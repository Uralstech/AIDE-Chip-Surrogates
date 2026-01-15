import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
import shap
import os

# -----------------------------
# Configuration
# -----------------------------
# List of all shard files
shard_files = [f"Data/results_shard_{i}.csv" for i in range(8)] 

# Output directory
output_dir = "surrogate_models_v2"
os.makedirs(output_dir, exist_ok=True)

# 1. Hyperparameter Tuning: 
#    - Reduced depth to 6 to prevent overfitting
#    - Lowered learning rate, increased estimators
#    - Added subsampling for robustness
xgb_params = {
    "n_estimators": 1500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": .8,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 100,
    "n_jobs": -1
}

xgb_objective_by_task = {
    "ipc": "reg:squarederror",
    "l2_miss_rate": "reg:absoluteerror",
}

# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
print("Loading and concatenating shards...")
dfs = []
for shard in shard_files:
    if os.path.exists(shard):
        dfs.append(pd.read_csv(shard))
    else:
        print(f"Warning: {shard} not found – skipping")

if not dfs:
    raise FileNotFoundError("No data files found.")

df = pd.concat(dfs, ignore_index=True)
df = df[df["error"] == 0].copy()
print(f"Valid rows loaded: {len(df)}")

# --- Feature Engineering ---

# 1. Log2 Transform Base Features
size_cols = ["l1d_size", "l1i_size", "l2_size"]
assoc_cols = ["l1d_assoc", "l1i_assoc", "l2_assoc"]

for col in size_cols + assoc_cols:
    df[f"{col}_log2"] = np.log2(df[col])

# 2. Derived Features (The "Cliff" Detectors)
# Ratio of L2 to L1D (Hierarchical inclusion clues)
df["l2_l1d_ratio_log2"] = df["l2_size_log2"] - df["l1d_size_log2"]

# Number of Sets proxy (Size - Assoc = Sets * LineSize). 
# Since LineSize is constant, this represents the log2(count of sets).
# This helps the model find conflict miss boundaries.
df["l1d_sets_log2"] = df["l1d_size_log2"] - df["l1d_assoc_log2"]
df["l2_sets_log2"] = df["l2_size_log2"] - df["l2_assoc_log2"]

# Define Final Feature List
feature_cols = (
    [f"{c}_log2" for c in size_cols] +
    [f"{c}_log2" for c in assoc_cols] +
    ["l2_l1d_ratio_log2", "l1d_sets_log2", "l2_sets_log2"]
)

print(f"Features used: {feature_cols}")

# Define Constraints (1 = Increasing, 0 = No constraint, -1 = Decreasing)
# Order must match feature_cols
# Sizes (pos), Assocs (neutral), Ratio (pos), Sets (pos)
# Note: For IPC, bigger is usually better. For Miss Rate, we handle logic below.
base_constraints = (1, 1, 1, 0, 0, 0, 1, 1, 1)

targets = ["ipc", "l2_miss_rate"]
workloads = sorted(df["workload"].unique())

# -----------------------------
# Training Loop
# -----------------------------
results_summary = []

for workload in workloads:
    print(f"\n=== Workload: {workload} ===")
    df_wl = df[df["workload"] == workload].copy()
    
    X = df_wl[feature_cols]
    y_all = df_wl[targets]

    # Split: Train (70%), Val (15%), Test (15%)
    X_train, X_temp, y_train_all, y_temp_all = train_test_split(
        X, y_all, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val_all, y_test_all = train_test_split(
        X_temp, y_temp_all, test_size=0.5, random_state=42
    )

    for target in targets:
        # Prepare Vectors
        y_train = y_train_all[target].values
        y_val = y_val_all[target].values
        y_test = y_test_all[target].values
        
        # 3. Log-Transform for Miss Rates
        # This fixes the "Orders of Magnitude" problem
        is_miss_rate_target = (target == "l2_miss_rate")
        
        if is_miss_rate_target:
            y_train_vec = np.log1p(y_train)
            y_val_vec = np.log1p(y_val)
        else:
            y_train_vec = y_train
            y_val_vec = y_val

        # 4. Dynamic Monotonic Constraints
        # Disable constraints for Dijkstra because associativity increases latency there
        if is_miss_rate_target:
            current_constraints = None
        else:
            current_constraints = base_constraints
            if workload in ["dijkstra", "fft", "sha"]:
                current_constraints = None

        # Configure Model
        model = XGBRegressor(
            monotone_constraints=current_constraints,
            objective=xgb_objective_by_task[target],
            **xgb_params
        )
        
        # Fit
        model.fit(
            X_train, y_train_vec,
            eval_set=[(X_val, y_val_vec)],
            verbose=False
        )
        
        # Predict
        pred_raw = model.predict(X_test)
        
        # Inverse Transform
        if is_miss_rate_target:
            pred = np.expm1(pred_raw)
            pred = np.clip(pred, 0, 1) # Clamp miss rates
        else:
            pred = pred_raw

        # 5. Calculate Metrics
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        mask = y_test > 1e-6
        smape = np.mean(
            np.abs(y_test - pred) /
            ((np.abs(y_test) + np.abs(pred)) / 2 + 1e-9)
        ) * 100

        p95_abs_err = np.percentile(np.abs(y_test - pred), 95)
        if is_miss_rate_target:
            p95_str = f"{p95_abs_err * 100:.2f}%"
        else:
            p95_str = f"{p95_abs_err:.3f} IPC"

        # Median Abs Error % (Robust against outliers)
        med_abs_err_pct = np.median(np.abs(y_test - pred) / np.clip(y_test, 1e-9, None)) * 100
        print(f"  [{target.upper()}] R²: {r2:.4f} | MAE: {mae:.4f} | SMAPE: {smape:.2f}% | P95 Abs Error: {p95_str} | MedAE%: {med_abs_err_pct:.3f}%")
        
        # Save Model
        joblib.dump(
            {"model": model, "features": feature_cols, "log_target": is_miss_rate_target},
            os.path.join(output_dir, f"model_{workload}_{target}.pkl")
        )

        # SHAP Plot (Use X_test subset to speed up if needed)
        # We calculate SHAP on the raw model output (log space if transformed)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP: {workload} - {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_{workload}_{target}.png"))
        plt.close()

        results_summary.append({
            "workload": workload,
            "target": target,
            "r2": r2,
            "mae": mae,
            "smape_%": smape,
            "p95_abs_err": p95_abs_err,
            "median_abs_error_%": med_abs_err_pct
        })

# -----------------------------
# Summary
# -----------------------------
summary_df = pd.DataFrame(results_summary)
summary_path = os.path.join(output_dir, "training_summary_v2.csv")
summary_df.to_csv(summary_path, index=False)

print("\n=== Final Training Summary ===")
# Display specific columns for readability
print(summary_df[["workload", "target", "r2", "smape_%", "p95_abs_err", "median_abs_error_%"]].round(4).to_string(index=False))
print(f"\nResults saved to {summary_path}")