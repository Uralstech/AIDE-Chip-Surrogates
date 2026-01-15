#!/usr/bin/env python3
"""
Surrogate Model Inference Script

Features:
- Workload alias handling (e.g. matrix → matrix_mul)
- Strict model existence checks (no silent fallback)
- Feature consistency validation
- Physical sanity checks on outputs
- Batch CSV inference
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_DIR = "surrogate_models_v2"

# Workload aliases (FIX for your issue)
WORKLOAD_ALIAS = {
    "matrix": "matrix_mul",
    "matmul": "matrix_mul",
}

TARGETS = ["ipc", "l2_miss_rate"]

# Expected feature order (MUST match training)
FEATURE_COLS = [
    "l1d_size_log2",
    "l1i_size_log2",
    "l2_size_log2",
    "l1d_assoc_log2",
    "l1i_assoc_log2",
    "l2_assoc_log2",
    "l2_l1d_ratio_log2",
    "l1d_sets_log2",
    "l2_sets_log2",
]

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def resolve_workload(workload: str) -> str:
    """Map aliases to canonical workload names"""
    return WORKLOAD_ALIAS.get(workload, workload)


def load_model(workload: str, target: str):
    """Load a model safely"""
    model_path = os.path.join(MODEL_DIR, f"model_{workload}_{target}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"    Check workload name or alias mapping."
        )
    payload = joblib.load(model_path)
    return payload["model"], payload["log_target"]


def physical_sanity_check(workload, ipc, miss_rate):
    """Detect physically impossible predictions"""
    warnings = []

    if ipc < 0 or ipc > 3.5:
        warnings.append(f"IPC={ipc:.3f} out of physical range")

    if miss_rate < 0 or miss_rate > 1:
        warnings.append(f"L2 miss rate={miss_rate:.3f} out of [0,1]")

    return warnings


# -------------------------------------------------
# Inference
# -------------------------------------------------
def run_inference(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_cols = [
        "workload",
        "l1d_size",
        "l1i_size",
        "l2_size",
        "l1d_assoc",
        "l1i_assoc",
        "l2_assoc",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # Feature Engineering (same as training)
    # -----------------------------
    for col in ["l1d_size", "l1i_size", "l2_size",
                "l1d_assoc", "l1i_assoc", "l2_assoc"]:
        df[f"{col}_log2"] = np.log2(df[col])

    df["l2_l1d_ratio_log2"] = df["l2_size_log2"] - df["l1d_size_log2"]
    df["l1d_sets_log2"] = df["l1d_size_log2"] - df["l1d_assoc_log2"]
    df["l2_sets_log2"] = df["l2_size_log2"] - df["l2_assoc_log2"]

    # Output columns
    df["pred_ipc"] = np.nan
    df["pred_l2_miss_rate"] = np.nan
    df["warnings"] = ""

    # -----------------------------
    # Row-wise inference
    # -----------------------------
    for idx, row in df.iterrows():
        raw_workload = row["workload"]
        workload = resolve_workload(raw_workload)

        X = row[FEATURE_COLS].values.reshape(1, -1)

        preds = {}
        warn_msgs = []

        for target in TARGETS:
            model, is_log_target = load_model(workload, target)

            pred_raw = model.predict(X)[0]
            pred = np.expm1(pred_raw) if is_log_target else pred_raw

            if target == "l2_miss_rate":
                pred = np.clip(pred, 0, 1)

            preds[target] = float(pred)

        warn_msgs.extend(
            physical_sanity_check(workload, preds["ipc"], preds["l2_miss_rate"])
        )

        df.at[idx, "pred_ipc"] = preds["ipc"]
        df.at[idx, "pred_l2_miss_rate"] = preds["l2_miss_rate"]
        df.at[idx, "warnings"] = "; ".join(warn_msgs)

    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to: {output_csv}")

    if (df["warnings"] != "").any():
        print("\n    Some rows triggered sanity warnings:")
        print(df[df["warnings"] != ""][["workload", "pred_ipc", "pred_l2_miss_rate", "warnings"]])


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_surrogate.py <input.csv> <output.csv>")
        sys.exit(1)

    run_inference(sys.argv[1], sys.argv[2])
