import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# -----------------------------
# Load data
# -----------------------------
sim_path = "simulation_results.csv"
pred_path = "predicted_results.csv"

sim_df = pd.read_csv(sim_path)
pred_df = pd.read_csv(pred_path)

# -----------------------------
# Normalize workload names (safety)
# -----------------------------
sim_df["workload"] = sim_df["workload"].str.lower()
pred_df["workload"] = pred_df["workload"].str.lower()

# -----------------------------
# Merge on configuration keys
# -----------------------------
key_cols = [
    "workload",
    "l1d_size", "l1i_size", "l2_size",
    "l1d_assoc", "l1i_assoc", "l2_assoc"
]

df = sim_df.merge(
    pred_df,
    on=key_cols,
    how="inner",
    suffixes=("_sim", "_pred")
)

print(f"Matched configurations: {len(df)}")

# -----------------------------
# Timing analysis
# -----------------------------
total_sim_time = sim_df["sim_duration_s"].sum()
critical_path_time = sim_df["sim_duration_s"].max()

# If inference time was measured externally
SURROGATE_TIME_SECONDS = 1.0764515399932861

print("\n=== Timing Comparison ===")
print(f"Total simulated CPU time     : {total_sim_time:,.2f} s")
print(f"Parallel critical-path time  : {critical_path_time:,.2f} s")
print(f"Surrogate inference time     : {SURROGATE_TIME_SECONDS:.3f} s")

print("\nSpeedups:")
print(f"  vs total simulation time   : {total_sim_time / SURROGATE_TIME_SECONDS:,.0f}×")
print(f"  vs critical-path time      : {critical_path_time / SURROGATE_TIME_SECONDS:,.0f}×")

# -----------------------------
# Accuracy metrics helper
# -----------------------------
def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100
    }

# -----------------------------
# Overall metrics
# -----------------------------
print("\n=== Overall Accuracy ===")

ipc_metrics = regression_metrics(df["ipc"], df["pred_ipc"])
miss_metrics = regression_metrics(df["l2_miss_rate"], df["pred_l2_miss_rate"])

print("\nIPC:")
for k, v in ipc_metrics.items():
    print(f"  {k:8}: {v:.6f}")

print("\nL2 Miss Rate:")
for k, v in miss_metrics.items():
    print(f"  {k:8}: {v:.6f}")

# -----------------------------
# Per-workload metrics
# -----------------------------
print("\n=== Per-Workload Accuracy ===")

rows = []

for wl, g in df.groupby("workload"):
    ipc_m = regression_metrics(g["ipc"], g["pred_ipc"])
    miss_m = regression_metrics(g["l2_miss_rate"], g["pred_l2_miss_rate"])

    rows.append({
        "workload": wl,
        "ipc_R2": ipc_m["R2"],
        "ipc_MAE": ipc_m["MAE"],
        "ipc_MAPE_%": ipc_m["MAPE_%"],
        "miss_R2": miss_m["R2"],
        "miss_MAE": miss_m["MAE"],
        "miss_MAPE_%": miss_m["MAPE_%"],
        "num_samples": len(g)
    })

summary_df = pd.DataFrame(rows).sort_values("workload")
print(summary_df.to_string(index=False, float_format="%.4f"))

# -----------------------------
# Save summary (optional)
# -----------------------------
summary_df.to_csv("model_vs_simulation_comparison.csv", index=False)
print("\nSaved: model_vs_simulation_comparison.csv")
