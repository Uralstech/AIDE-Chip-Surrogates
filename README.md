# AIDE-Chip Surrogates
**Fast, Explainable Surrogate Models for gem5 Cache Design Space Exploration**

This repository contains the complete pipeline for generating, training, validating, and deploying physics-aware machine-learning surrogates that replace most gem5 cache simulations during microarchitecture design-space exploration (DSE).

Using ~15,000 validated gem5 SE-mode simulations across six workloads, we train monotonicity-constrained XGBoost models that predict IPC and L2 miss rate from cache configuration parameters. The resulting models achieve near-perfect accuracy on held-out data and deliver 800×+ speedups over cycle-accurate simulation on out-of-distribution (OOD) cache configurations.

## Preprint

Udayshankar Ravikumar . Fast, Explainable Surrogate Models for gem5 Cache Design Space Exploration. Authorea. January 14, 2026.
https://doi.org/10.22541/au.176843174.46109183/v1

## Repository Structure

```
AIDE-CHIP-SURROGATES/
├── 15k-Sims/                   # Dataset generation using gem5
│   ├── bash/                   # AWS multi-node launch scripts
│   ├── benchmarks/             # crc32, dijkstra, fft, matrix_mul, qsort, sha
│   ├── config-gen/             # Cache configuration generator
│   ├── requirements.txt
│   └── run_simulations.py
├── train.py                    # Train surrogate models
├── inference.py                # Run fast predictions
├── ood-compare/                # 26-config OOD stress test
│   ├── predicted_results.csv
│   ├── simulation_results.csv
│   └── compare.py              # Compares the contents of predicted_results.csv and simulation_results.csv by time taken and various error metrics
└── requirements.txt
```

## Workloads

Six benchmarks were chosen to span diverse cache behaviors:

| Workload | Behavior |
|--------|--------|
| crc32 | streaming, low locality |
| dijkstra | pointer-chasing, irregular |
| fft | strided, cache-sensitive |
| matrix_mul | dense compute, high reuse |
| qsort | branchy, mixed locality |
| sha | compute-bound, near-zero miss rate |

## Dataset Generation

All simulations were run using gem5 SE-mode (Syscall Emulation), enabling fast and deterministic cache-accurate simulation of short kernels.

Simulations were executed on:
- 4× AWS c6g + 4× AWS c7g
- 64 vCPUs each
- 8-node parallel execution

Total dataset size: ~15,000 valid cache configurations

## Training the Surrogates

`train.py` trains 12 models (6 workloads × IPC & L2 miss rate) using:

* Log-transformed cache sizes & associativities
* Set-count and hierarchy-ratio features
* Monotonic constraints enforcing cache physics
* SHAP explainability artifacts

Training time: < 1 minute on CPU

## Fast Inference

`inference.py`:

* Loads the correct workload-specific model
* Applies feature engineering
* Produces IPC & L2 miss predictions
* Flags physical violations (IPC < 0, miss > 1, etc.)

## OOD Validation

We validate on 26 out-of-distribution cache configs:

`compare.py` compares:

* gem5 ground truth
* surrogate predictions
* wall-clock time vs simulation time

Measured:

* **879s gem5 wall-clock**
* **1.07s surrogate inference**
* **~817× critical-path speedup**
