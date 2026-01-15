import csv
from itertools import product
import random

CACHELINE = 64  # bytes
TOTAL_REQUIRED = 15_000

def valid_assocs(size_kb, max_assoc):
    lines = (size_kb * 1024) // CACHELINE
    return [a for a in [1, 2, 4, 8, 16, 32] if a <= max_assoc and lines % a == 0]

L1_SIZES = [16, 32, 64, 128]
L2_SIZES = [128, 256, 512, 1024]

WORKLOADS = ["matrix_mul", "dijkstra", "fft", "qsort", "crc32", "sha"]

# ------------------------------------------------------------------
# 1. Generate ALL workload-independent cache configurations
# ------------------------------------------------------------------
base_configs = []

for l1d_size, l1i_size, l2_size in product(L1_SIZES, L1_SIZES, L2_SIZES):
    for l1d_assoc in valid_assocs(l1d_size, 8):
        for l1i_assoc in valid_assocs(l1i_size, 8):
            for l2_assoc in valid_assocs(l2_size, 16):
                base_configs.append({
                    "l1d_size": l1d_size,
                    "l1i_size": l1i_size,
                    "l2_size": l2_size,
                    "l1d_assoc": l1d_assoc,
                    "l1i_assoc": l1i_assoc,
                    "l2_assoc": l2_assoc,
                })

random.shuffle(base_configs)

# ------------------------------------------------------------------
# 2. Truncate to equal share per workload
# ------------------------------------------------------------------
per_workload = TOTAL_REQUIRED // len(WORKLOADS)
base_configs = base_configs[:per_workload]

# ------------------------------------------------------------------
# 3. Duplicate configs for each workload
# ------------------------------------------------------------------
rows = []
for workload in WORKLOADS:
    for cfg in base_configs:
        row = cfg.copy()
        row["workload"] = workload
        rows.append(row)

# ------------------------------------------------------------------
# 4. Write CSV
# ------------------------------------------------------------------
with open("configs_15k.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} configs ({per_workload} per workload)")
