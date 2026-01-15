#!/usr/bin/env python3
"""
gem5 Simulation Executor for Chip Design Space Exploration
PRODUCTION-READY VERSION with all fixes applied

Features:
- Workload-specific timeouts (up to 14400s)
- Benchmark validation before execution
- Instance-ID hash-based work sharding (no duplication)
- Auto-shutdown on idle (1 hour)
- S3 heartbeat tracking (every 5 minutes)
- Retry logic (2 attempts for transient failures)
- Result sanity validation
- Disk space monitoring
- Graceful spot interruption handling
- Comprehensive logging and error tracking
"""

import os
import sys
import subprocess
import pandas as pd
import re
import signal
import shutil
import json
import hashlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Try importing boto3
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("⚠️  Warning: boto3 not available. S3 sync disabled.")

# ============================================
# LOGGING SETUP
# ============================================
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = os.getenv('GEM5_DIR', '/home/ubuntu/gem5')
PROJECT_DIR = os.getenv('PROJECT_DIR', '/home/ubuntu/')

# gem5 executable
GEM5_EXECUTABLE = os.path.join(BASE_DIR, "build", "RISCV", "gem5.opt")

# gem5 config script - try both paths
CONFIG_PATHS = [
    os.path.join(BASE_DIR, "configs", "deprecated", "example", "se.py"),
    os.path.join(BASE_DIR, "configs", "example", "se.py")
]

GEM5_CONFIG_SCRIPT = None  # Will be set in validate_gem5_setup()

BENCHMARK_DIR = os.path.join(PROJECT_DIR, "compiled")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# S3 configuration
S3_BUCKET = os.getenv('S3_BUCKET', None)
S3_REGION = os.getenv('AWS_REGION', 'ap-south-2')
S3_ENABLED = S3_BUCKET is not None and BOTO3_AVAILABLE

# Work sharding configuration
TOTAL_INSTANCES = int(os.getenv('TOTAL_INSTANCES', '1'))
INSTANCE_INDEX = int(os.getenv('INSTANCE_INDEX', '0'))
INSTANCE_TAG = f"shard_{INSTANCE_INDEX}"

# Workload-specific timeouts (in seconds)
WORKLOAD_TIMEOUTS = {
    'matrix_mul': 7200,    # 2 hours - 256x256 matrix
    'dijkstra': 14400,     # 4 hours - N=1000 graph (complex)
    'crc32': 5400,         # 1.5 hours - 1MB data
    'fft': 14400,          # 4 hours - N=4096 FFT (recursive)
    'qsort': 7200,         # 2 hours - 100K elements
    'sha': 10800           # 3 hours - 10K iterations
}
DEFAULT_TIMEOUT = 14400  # 4 hours for unknown workloads

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 10  # seconds

# Auto-shutdown configuration
IDLE_THRESHOLD = 3600  # 1 hour of no activity

# Heartbeat configuration
HEARTBEAT_INTERVAL = 300  # 5 minutes

# Graceful shutdown flag
shutdown_flag = False

# Global counters for tracking
completed_count = 0
failed_count = 0
last_heartbeat = time.time()
last_cleanup = time.time()

# ============================================
# SIGNAL HANDLERS (Spot Interruption)
# ============================================
def handle_termination(signum, frame):
    """Handle SIGTERM/SIGINT from spot interruption monitor"""
    global shutdown_flag
    shutdown_flag = True
    logger.warning("⚠️  TERMINATION SIGNAL RECEIVED - Initiating graceful shutdown...")
    logger.warning("⚠️  Current simulations will complete, new ones will be cancelled")

signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)

# ============================================
# HELPER FUNCTIONS - VALIDATION
# ============================================
def validate_gem5_setup():
    """Verify gem5 is properly set up"""
    global GEM5_CONFIG_SCRIPT
    
    # Check gem5 executable exists
    if not os.path.exists(GEM5_EXECUTABLE):
        logger.error(f"❌ gem5 executable not found: {GEM5_EXECUTABLE}")
        logger.error("Run setup_and_run.sh first to compile gem5")
        return False
    
    # Check size - should be ~180MB
    try:
        gem5_size_mb = os.path.getsize(GEM5_EXECUTABLE) / (1024 * 1024)
        if gem5_size_mb < 100:
            logger.error(f"❌ gem5 binary too small ({gem5_size_mb:.1f}MB)")
            logger.error("   This indicates incomplete or phantom compilation")
            logger.error("   Expected: ~180MB")
            return False
        logger.info(f"✓ gem5 executable verified ({gem5_size_mb:.1f}MB)")
    except Exception as e:
        logger.error(f"❌ Error checking gem5 binary: {e}")
        return False
    
    # Find config script
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            GEM5_CONFIG_SCRIPT = path
            logger.info(f"✓ gem5 config script: {path}")
            break
    
    if GEM5_CONFIG_SCRIPT is None:
        logger.error(f"❌ gem5 config script not found in any of:")
        for path in CONFIG_PATHS:
            logger.error(f"   - {path}")
        return False
    
    return True

def validate_benchmarks():
    """Verify all benchmarks are compiled and executable"""
    required_benchmarks = ['matrix_mul', 'dijkstra', 'crc32', 'fft', 'qsort', 'sha']
    missing = []
    invalid = []
    
    logger.info("Validating benchmarks...")
    
    for bench in required_benchmarks:
        bench_path = os.path.join(BENCHMARK_DIR, f"{bench}.riscv")
        
        if not os.path.exists(bench_path):
            missing.append(bench)
            continue
        
        # Check if executable and reasonable size
        try:
            size_kb = os.path.getsize(bench_path) / 1024
            
            # Benchmarks should be at least 50KB (statically linked)
            if size_kb < 50:
                invalid.append(f"{bench} ({size_kb:.1f}KB - too small)")
                continue
            
            # Check if executable bit is set
            if not os.access(bench_path, os.X_OK):
                invalid.append(f"{bench} (not executable)")
                continue
            
            logger.info(f"  ✓ {bench:12} ({size_kb:7.1f} KB)")
            
        except Exception as e:
            invalid.append(f"{bench} ({e})")
    
    if missing:
        logger.error(f"❌ Missing benchmarks: {missing}")
        logger.error("   Run benchmark compilation script first")
        return False
    
    if invalid:
        logger.error(f"❌ Invalid benchmarks: {invalid}")
        logger.error("   Recompile benchmarks with proper flags:")
        logger.error("   riscv64-linux-gnu-gcc -O3 -static -march=rv64gc -o bench.riscv bench.c -lm")
        return False
    
    logger.info(f"✓ All {len(required_benchmarks)} benchmarks validated")
    return True

def validate_s3_access():
    """Validate S3 access before starting simulations"""
    global S3_ENABLED
    
    if not S3_ENABLED:
        logger.warning("⚠️  S3 sync disabled (boto3 not available or S3_BUCKET not set)")
        return True
    
    try:
        s3 = boto3.client('s3', region_name=S3_REGION)
        # Try to head bucket (validates credentials and bucket exists)
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ S3 access validated: s3://{S3_BUCKET} (region: {S3_REGION})")
        return True
    except Exception as e:
        logger.error(f"❌ S3 access failed: {e}")
        logger.warning("⚠️  Continuing without S3 sync - results will only be saved locally")
        S3_ENABLED = False
        return True  # Don't fail, just disable S3

# ============================================
# HELPER FUNCTIONS - CONFIGURATION
# ============================================
def is_power_of_two(x):
    return x > 0 and (x & (x - 1) == 0)

def is_valid_config(config):
    """Check if cache sizes and associativities are valid"""
    
    # Check associativity (must be power of two)
    for key in ['l1d_assoc', 'l1i_assoc', 'l2_assoc']:
        assoc = int(config.get(key, 0))
        if assoc <= 0 or not is_power_of_two(assoc):
            return False
    
    # Check cache sizes (must be power of two, in KB)
    for key in ['l1d_size', 'l1i_size', 'l2_size']:
        size_kb = parse_cache_size(config.get(key))
        if size_kb <= 0 or not is_power_of_two(size_kb):
            return False
    
    return True

def parse_cache_size(size_str):
    """Convert cache size string to integer KB"""
    size_str = str(size_str).strip().upper()
    
    # Handle string formats with units
    if 'KB' in size_str:
        return int(size_str.replace('KB', '').replace('K', ''))
    elif 'MB' in size_str:
        return int(size_str.replace('MB', '').replace('M', '')) * 1024
    elif 'GB' in size_str:
        return int(size_str.replace('GB', '').replace('G', '')) * 1024 * 1024
    else:
        # Assume it's already an integer in KB
        try:
            return int(float(size_str))
        except:
            logger.error(f"Cannot parse cache size: {size_str}")
            return 64  # Default fallback

# ============================================
# HELPER FUNCTIONS - RESULTS PARSING
# ============================================
def parse_stats(stats_file):
    """Extract IPC and L2 miss rate from gem5 stats.txt"""
    stats = {'ipc': None, 'l2_miss_rate': None}
    
    if not os.path.exists(stats_file):
        logger.warning(f"Stats file not found: {stats_file}")
        return stats
    
    try:
        with open(stats_file, 'r') as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read stats file: {e}")
        return stats
    
    # Flexible regex for floating point numbers
    fp_regex = r'([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)'
    
    # Extract IPC
    ipc_match = re.search(r'system\.cpu\.ipc\s+' + fp_regex, content)
    if ipc_match:
        stats['ipc'] = float(ipc_match.group(1))
    
    # Extract L2 miss rate
    l2_match = re.search(r'system\.l2\.overallMissRate::total\s+' + fp_regex, content)
    if l2_match:
        stats['l2_miss_rate'] = float(l2_match.group(1))
    
    return stats

def validate_results(stats, config, duration):
    """Validate simulation results are physically reasonable"""
    ipc = stats.get('ipc')
    l2_miss = stats.get('l2_miss_rate')
    
    # IPC should be between 0.01 and 8.0 for O3CPU
    if ipc is not None:
        if ipc < 0.01:
            return False, f"IPC too low: {ipc} (likely simulation crashed early)"
        if ipc > 8.0:
            return False, f"IPC too high: {ipc} (unrealistic for O3CPU)"
    
    # L2 miss rate should be between 0 and 1
    if l2_miss is not None:
        if l2_miss < 0 or l2_miss > 1:
            return False, f"L2 miss rate out of range: {l2_miss}"
    
    # ⚠️ DO NOT check duration anymore
    return True, None

# ============================================
# HELPER FUNCTIONS - S3 OPERATIONS
# ============================================
def sync_to_s3(local_file, s3_path, retries=3):
    """Upload file to S3 with retry logic"""
    if not S3_ENABLED:
        return False
    
    for attempt in range(retries):
        try:
            s3 = boto3.client('s3', region_name=S3_REGION)
            s3.upload_file(local_file, S3_BUCKET, s3_path)
            logger.info(f"✓ Synced to S3: s3://{S3_BUCKET}/{s3_path}")
            return True
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"S3 sync failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(5)
            else:
                logger.error(f"❌ S3 sync failed after {retries} attempts: {e}")
                return False
    return False

def update_heartbeat(output_file):
    """Update heartbeat file in S3 to track instance health"""
    global last_heartbeat
    
    if not S3_ENABLED:
        return
    
    # Only update every HEARTBEAT_INTERVAL seconds
    if time.time() - last_heartbeat < HEARTBEAT_INTERVAL:
        return
    
    try:
        instance_id = os.getenv('INSTANCE_ID', 'unknown')
        
        heartbeat_data = {
            'instance_id': instance_id,
            'timestamp': time.time(),
            'timestamp_human': datetime.now().isoformat(),
            'completed': completed_count,
            'failed': failed_count,
            'success_rate': f"{100 * completed_count / max(1, completed_count + failed_count):.1f}%",
            'status': 'running',
            'output_file': output_file
        }
        
        heartbeat_file = f"/tmp/heartbeat_{INSTANCE_TAG}.json"
        with open(heartbeat_file, 'w') as f:
            json.dump(heartbeat_data, f, indent=2)
        
        sync_to_s3(heartbeat_file, f"heartbeat/{INSTANCE_TAG}.json")
        last_heartbeat = time.time()
        
    except Exception as e:
        logger.error(f"Error updating heartbeat: {e}")

# ============================================
# HELPER FUNCTIONS - MAINTENANCE
# ============================================
def cleanup_old_m5out():
    """Clean up old m5out directories to prevent disk full"""
    global last_cleanup

    # Only cleanup every 30 minutes
    if time.time() - last_cleanup < 1800:
        return

    m5out_base = os.path.join(BASE_DIR, "m5out")

    # If m5out doesn't exist, nothing to do
    if not os.path.exists(m5out_base):
        logger.debug("m5out directory not found, creating it.")
        try:
            os.makedirs(m5out_base, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create m5out base dir: {e}")
            return
    
    try:
        # Get disk usage
        stat = os.statvfs(m5out_base)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        used_pct = 100 * (1 - stat.f_bavail / stat.f_blocks)
        
        logger.info(f"Disk usage: {used_pct:.1f}% ({free_gb:.1f}GB free of {total_gb:.1f}GB)")
        
        # Clean if less than 10GB free OR more than 90% used
        if free_gb < 10 or used_pct > 90:
            logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free - cleaning old m5out...")
            
            # Get all m5out subdirectories sorted by modification time
            dirs = sorted(Path(m5out_base).iterdir(), key=os.path.getmtime, reverse=True)
            
            # Keep only most recent 100 directories
            removed_count = 0
            for old_dir in dirs[100:]:
                if old_dir.is_dir():
                    try:
                        shutil.rmtree(old_dir)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {old_dir}: {e}")
            
            if removed_count > 0:
                logger.info(f"✓ Cleaned {removed_count} old m5out directories")
        
        last_cleanup = time.time()
        
    except Exception as e:
        logger.error(f"Error cleaning m5out: {e}")

def check_idle_and_shutdown(output_file):
    """Check if simulations are done and initiate shutdown"""
    try:
        # Check if any gem5 processes running
        result = subprocess.run(['pgrep', '-f', 'gem5.opt'], 
                               capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:  # No gem5 processes
            logger.info("No gem5 processes detected - checking idle time...")
            
            # Check if results file hasn't been modified in IDLE_THRESHOLD
            if os.path.exists(output_file):
                mod_time = os.path.getmtime(output_file)
                idle_seconds = time.time() - mod_time
                
                if idle_seconds > IDLE_THRESHOLD:
                    logger.warning(f"⚠️  IDLE for {idle_seconds/3600:.1f} hours - initiating shutdown")
                    
                    # Final S3 sync
                    if S3_ENABLED:
                        logger.info("Performing final S3 sync before shutdown...")
                        sync_to_s3(output_file, f"results/{Path(output_file).name}")
                        
                        # Update heartbeat with final status
                        instance_id = os.getenv('INSTANCE_ID', 'unknown')
                        heartbeat_data = {
                            'instance_id': instance_id,
                            'timestamp': time.time(),
                            'timestamp_human': datetime.now().isoformat(),
                            'completed': completed_count,
                            'failed': failed_count,
                            'status': 'completed',
                            'shutdown_reason': 'idle_timeout'
                        }
                        heartbeat_file = f"/tmp/heartbeat_{INSTANCE_TAG}.json"
                        with open(heartbeat_file, 'w') as f:
                            json.dump(heartbeat_data, f, indent=2)
                        sync_to_s3(heartbeat_file, f"heartbeat/{INSTANCE_TAG}.json")
                    
                    # Terminate instance (only works on EC2)
                    instance_id = os.getenv('INSTANCE_ID')
                    if instance_id and instance_id.startswith('i-'):
                        logger.info(f"Terminating instance {instance_id}...")
                        subprocess.run(['sudo', 'shutdown', '-h', '+1'], check=False)
                        return True
                    else:
                        logger.info("Not on EC2 - skipping auto-shutdown")
                else:
                    logger.info(f"Idle for {idle_seconds:.0f}s (threshold: {IDLE_THRESHOLD}s)")
    
    except Exception as e:
        logger.error(f"Error checking idle status: {e}")
    
    return False

# ============================================
# SIMULATION EXECUTION
# ============================================
def run_simulation(config):
    """Run a single gem5 simulation"""
    
    # Check shutdown flag
    if shutdown_flag:
        logger.info("Skipping simulation due to shutdown signal")
        return None
    
    # Validate config
    if not is_valid_config(config):
        logger.warning(f"Invalid config (non-power-of-2 associativity): {config}")
        return {
            **config, 
            'ipc': None, 
            'l2_miss_rate': None, 
            'sim_duration_s': 0, 
            'error': 3, 
            'error_msg': 'Invalid config: cache size or associativity not power of 2'
        }
    
    # Get workload
    workload = config.get('workload', 'matrix_mul')
    benchmark_path = os.path.join(BENCHMARK_DIR, f"{workload}.riscv")
    
    if not os.path.exists(benchmark_path):
        logger.error(f"Benchmark not found: {benchmark_path}")
        return {
            **config, 
            'ipc': None, 
            'l2_miss_rate': None, 
            'sim_duration_s': 0, 
            'error': 4, 
            'error_msg': f'Benchmark not found: {workload}.riscv'
        }
    
    # Parse cache sizes
    l1d_size = parse_cache_size(config['l1d_size'])
    l1i_size = parse_cache_size(config['l1i_size'])
    l2_size = parse_cache_size(config['l2_size'])
    
    # Create unique run ID
    run_id = (f"l1d_{l1d_size}_{config['l1d_assoc']}_"
              f"l1i_{l1i_size}_{config['l1i_assoc']}_"
              f"l2_{l2_size}_{config['l2_assoc']}_"
              f"{workload}")
    
    output_dir = os.path.join(BASE_DIR, "m5out", run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build gem5 command
    command = [
        GEM5_EXECUTABLE,
        f"--outdir={output_dir}",
        GEM5_CONFIG_SCRIPT,
        "--cpu-type=O3CPU",
        "--caches",
        "--l2cache",
        "--num-cpus=1",
        "--mem-type=SimpleMemory",
        f"--l1d_size={l1d_size}kB",
        f"--l1i_size={l1i_size}kB",
        f"--l2_size={l2_size}kB",
        f"--l1d_assoc={config['l1d_assoc']}",
        f"--l1i_assoc={config['l1i_assoc']}",
        f"--l2_assoc={config['l2_assoc']}",
        "-c", benchmark_path
    ]
    
    # Get workload-specific timeout
    timeout = WORKLOAD_TIMEOUTS.get(workload, DEFAULT_TIMEOUT)
    
    # Run simulation
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        duration = time.time() - start_time
        
        # Do NOT trust return code for gem5 SE mode
        stats_file = os.path.join(output_dir, "stats.txt")
        
        if not os.path.exists(stats_file):
            error_msg = (result.stderr or result.stdout or "No output")[:500]
            logger.error(
                f"gem5 produced no stats.txt for {run_id} "
                f"(rc={result.returncode})"
            )
            return {
                **config,
                'ipc': None,
                'l2_miss_rate': None,
                'sim_duration_s': round(duration, 2),
                'error': 1,
                'error_msg': error_msg
            }


        # Parse results
        stats = parse_stats(os.path.join(output_dir, "stats.txt"))
        
        # Validate IPC was extracted
        if stats['ipc'] is None:
            return {
                **config,
                **stats,
                'sim_duration_s': round(duration, 2),
                'error': 6,
                'error_msg': 'Failed to extract IPC from stats.txt'
            }
        
        # Validate results are physically reasonable
        valid, reason = validate_results(stats, config, duration)
        if not valid:
            return {
                **config,
                **stats,
                'sim_duration_s': round(duration, 2),
                'error': 7,
                'error_msg': f'Invalid results: {reason}'
            }
        
        return {
            **config,
            **stats,
            'sim_duration_s': round(duration, 2),
            'error': 0,
            'error_msg': None
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Timeout for {run_id} after {duration:.1f}s (limit: {timeout}s)")
        return {
            **config, 
            'ipc': None, 
            'l2_miss_rate': None,
            'sim_duration_s': round(duration, 2), 
            'error': 2,
            'error_msg': f'Timeout (>{timeout}s)'
        }
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)[:500]
        logger.error(f"Unexpected error for {run_id}: {error_msg}")
        return {
            **config, 
            'ipc': None, 
            'l2_miss_rate': None,
            'sim_duration_s': round(duration, 2), 
            'error': 5,
            'error_msg': error_msg
        }

def run_simulation_with_retry(config):
    """Run simulation with retry logic for transient failures"""
    for attempt in range(MAX_RETRIES + 1):
        result = run_simulation(config)
        
        # Retry on specific error codes (1=gem5 error, 5=unexpected)
        if result and result.get('error') in [1, 5] and attempt < MAX_RETRIES:
            workload = config.get('workload', 'unknown')
            logger.warning(f"Retry {attempt+1}/{MAX_RETRIES} for {workload} config")
            time.sleep(RETRY_DELAY)
            continue
        
        return result
    
    return result

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run gem5 simulations for chip design space exploration"
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to CSV file with configurations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--sync-interval', type=int, default=100,
                       help='Sync to S3 every N simulations')
    
    args = parser.parse_args()
    
    # ============================================
    # STARTUP VALIDATION
    # ============================================
    logger.info("="*60)
    logger.info("gem5 SIMULATION EXECUTOR - PRODUCTION VERSION")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    # Validate gem5 setup
    if not validate_gem5_setup():
        logger.error("❌ gem5 setup validation failed")
        sys.exit(1)
    
    # Validate benchmarks
    if not validate_benchmarks():
        logger.error("❌ Benchmark validation failed")
        sys.exit(1)
    
    # Validate S3 access
    validate_s3_access()
    logger.info(f"Using fixed INSTANCE_INDEX={INSTANCE_INDEX} (manual sharding)")
    
    # Log configuration
    logger.info(f"Work sharding: Instance {INSTANCE_INDEX}/{TOTAL_INSTANCES}")
    logger.info(f"S3 sync: {'Enabled' if S3_ENABLED else 'Disabled'}")
    if S3_ENABLED:
        logger.info(f"S3 bucket: s3://{S3_BUCKET} (region: {S3_REGION})")
    logger.info(f"Timeout configuration: {WORKLOAD_TIMEOUTS}")
    logger.info(f"Retry logic: {MAX_RETRIES} retries with {RETRY_DELAY}s delay")
    logger.info(f"Auto-shutdown: Enabled (idle threshold: {IDLE_THRESHOLD}s)")
    logger.info(f"Heartbeat: Every {HEARTBEAT_INTERVAL}s")
    
    # Set output file
    if args.output is None:
        args.output = os.path.join(RESULTS_DIR, f"results_{INSTANCE_TAG}.csv")
    
    logger.info(f"Output file: {args.output}")
    
    # ============================================
    # LOAD CONFIGURATIONS
    # ============================================
    logger.info(f"Loading configurations from {args.config}")
    
    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    configs_df = pd.read_csv(args.config)
    logger.info(f"✓ Loaded {len(configs_df)} total configurations")
    
    # Validate required columns
    required_cols = ['l1d_size', 'l1i_size', 'l2_size', 
                     'l1d_assoc', 'l1i_assoc', 'l2_assoc', 'workload']
    missing_cols = set(required_cols) - set(configs_df.columns)
    if missing_cols:
        logger.error(f"❌ Missing columns in config file: {missing_cols}")
        logger.error(f"Available columns: {list(configs_df.columns)}")
        sys.exit(1)
    
    # ============================================
    # WORK SHARDING
    # ============================================
    if TOTAL_INSTANCES > 1:
        logger.info(f"Applying work sharding for instance {INSTANCE_INDEX}/{TOTAL_INSTANCES}")
        configs_df['_config_index'] = range(len(configs_df))
        configs_df = configs_df[configs_df['_config_index'] % TOTAL_INSTANCES == INSTANCE_INDEX]
        configs_df = configs_df.drop('_config_index', axis=1).reset_index(drop=True)
        logger.info(f"✓ After sharding: {len(configs_df)} configs assigned to this instance")
        logger.info(f"  This instance will process configs: {INSTANCE_INDEX}, {INSTANCE_INDEX + TOTAL_INSTANCES}, {INSTANCE_INDEX + 2*TOTAL_INSTANCES}, ...")
    
    # ============================================
    # RESUME CAPABILITY
    # ============================================
    if os.path.exists(args.output):
        results_df = pd.read_csv(args.output)
        logger.info(f"✓ Found existing results: {len(results_df)} rows")
        
        # Identify only SUCCESSFUL configs
        successful = set(
            results_df[results_df['error'] == 0][required_cols]
            .apply(tuple, axis=1)
        )

        configs_to_run = configs_df[
            ~configs_df[required_cols].apply(tuple, axis=1).isin(successful)
        ]

        logger.info(f"✓ Successful configs already completed: {len(successful)}")
        logger.info(f"✓ Configs to (re)run: {len(configs_to_run)}")

        logger.info(f"✓ Remaining configs to run: {len(configs_to_run)}")
    else:
        results_df = pd.DataFrame()
        configs_to_run = configs_df
        logger.info("No previous results found - starting fresh")
    
    if len(configs_to_run) == 0:
        logger.info("✅ All simulations already complete!")
        sys.exit(0)
    
    configs_list = configs_to_run.to_dict('records')
    
    # Set number of workers
    if args.workers is None:
        args.workers = min(max(1, os.cpu_count() - 2), 56)  # don't spawn more than 56 workers by default
    logger.info(f"Workers: {args.workers} parallel processes")
    
    logger.info("="*60)
    logger.info(f"STARTING {len(configs_list)} SIMULATIONS")
    logger.info("="*60)
    
    # ============================================
    # RUN SIMULATIONS
    # ============================================
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_simulation_with_retry, config): config 
            for config in configs_list
        }
        
        # Process as they complete
        for future in tqdm(as_completed(future_to_config), 
                          total=len(configs_list), 
                          desc="Simulations"):
            
            # Check shutdown flag
            if shutdown_flag:
                logger.warning("⚠️  Shutdown flag set - cancelling remaining jobs...")
                for f in future_to_config:
                    f.cancel()
                break
            
            try:
                row = future.result()
                
                if row is not None:
                    # --------------------------------------------
                    # REMOVE any previous entry for this config
                    # (important when rerunning failed cases)
                    # --------------------------------------------
                    if not results_df.empty:
                        results_df = results_df[
                            ~(
                                (results_df['l1d_size'] == row['l1d_size']) &
                                (results_df['l1i_size'] == row['l1i_size']) &
                                (results_df['l2_size'] == row['l2_size']) &
                                (results_df['l1d_assoc'] == row['l1d_assoc']) &
                                (results_df['l1i_assoc'] == row['l1i_assoc']) &
                                (results_df['l2_assoc'] == row['l2_assoc']) &
                                (results_df['workload'] == row['workload'])
                            )
                        ]
                
                    # --------------------------------------------
                    # Append fresh result
                    # --------------------------------------------
                    results_df = pd.concat(
                        [results_df, pd.DataFrame([row])],
                        ignore_index=True
                    )
                
                    # Save immediately (fault tolerance)
                    results_df.to_csv(args.output, index=False)
                
                    
                    completed_count += 1
                    if row.get('error', 0) != 0:
                        failed_count += 1
                    
                    # Periodic S3 sync
                    if S3_ENABLED and completed_count % args.sync_interval == 0:
                        s3_path = f"results/results_{INSTANCE_TAG}.csv"
                        sync_to_s3(args.output, s3_path)
                    
                    # Update heartbeat
                    update_heartbeat(args.output)
                    
                    # Cleanup old m5out directories
                    cleanup_old_m5out()
                        
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                failed_count += 1
    
    # ============================================
    # FINALIZATION
    # ============================================
    # Final sync
    if S3_ENABLED:
        logger.info("Performing final S3 sync...")
        s3_path = f"results/results_{INSTANCE_TAG}.csv"
        sync_to_s3(args.output, s3_path)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*60)
    logger.info(f"End time: {datetime.now().isoformat()}")
    
    final_df = pd.read_csv(args.output)
    logger.info(f"Total simulations: {len(final_df)}")
    
    if 'error' in final_df.columns:
        success = len(final_df[final_df['error'] == 0])
        failed = len(final_df[final_df['error'] != 0])
        logger.info(f"Successful: {success} ({100*success/len(final_df):.1f}%)")
        logger.info(f"Failed: {failed} ({100*failed/len(final_df):.1f}%)")
        
        if failed > 0:
            error_counts = final_df[final_df['error'] != 0]['error'].value_counts()
            logger.info("\nError breakdown:")
            error_names = {
                1: "gem5 execution error",
                2: "Timeout",
                3: "Invalid config",
                4: "Benchmark not found",
                5: "Unexpected error",
                6: "Failed to extract IPC",
                7: "Invalid results (sanity check failed)"
            }
            for error_code, count in error_counts.items():
                logger.info(f"  {error_names.get(error_code, f'Error {error_code}')}: {count}")
    
    logger.info(f"\nResults saved to: {args.output}")
    if S3_ENABLED:
        logger.info(f"Results synced to: s3://{S3_BUCKET}/results/")
    
    logger.info("="*60)
    
    # Check for auto-shutdown
    logger.info("\nChecking for auto-shutdown conditions...")
    check_idle_and_shutdown(args.output)