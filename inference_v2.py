import sys, os, time, json, pandas, numpy, xgboost

MODEL_DIR = "surrogate_models_v2_json"
WORKLOADS = {
    "crc32",
    "dijkstra",
    "fft",
    "qsort",
    "sha",
    "matrix_mul",
}

TARGETS = {
    "ipc",
    "l2_miss_rate"
}

REQUIRED_INPUT_COLUMNS = {
    "l1d_size",
    "l1i_size",
    "l2_size",
    "l1d_assoc",
    "l1i_assoc",
    "l2_assoc",
    "workload",
}

FEATURES = [
    "l1d_size_log2",
    "l1i_size_log2",
    "l2_size_log2",
    "l1d_assoc_log2",
    "l1i_assoc_log2",
    "l2_assoc_log2",
    "l2_l1d_ratio_log2",
    "l1d_sets_log2",
    "l2_sets_log2"
]

def load_models() -> dict[str, (xgboost.XGBRegressor, bool)]:
    
    start_time = time.perf_counter()
    models = { }

    for workload in WORKLOADS:
        for target in TARGETS:
            model_name = f"model_{workload}_{target}"

            model_file = os.path.join(MODEL_DIR, f"{model_name}.json")
            metadata_file = os.path.join(MODEL_DIR, f"{model_name}_meta.json")

            regressor = xgboost.XGBRegressor()
            regressor.load_model(model_file)

            with open(metadata_file, 'r') as file:
                metadata = json.load(file)

            models[model_name] = (regressor, metadata['log_target'])

    elapsed = time.perf_counter() - start_time
    print(f"⏱️  Model loading time: {elapsed:.3f} sec")

    return models

def physical_sanity_check(workload, ipc, miss_rate):
    """Detect physically impossible predictions"""
    warnings = []

    if ipc < 0 or ipc > 3.5:
        warnings.append(f"IPC={ipc:.3f} out of physical range")

    if miss_rate < 0 or miss_rate > 1:
        warnings.append(f"L2 miss rate={miss_rate:.3f} out of [0,1]")

    return warnings

def run_inference(models, input_path, output_path):

    df = pandas.read_csv(input_path)

    missing_cols = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if (missing_cols):
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    start_time = time.perf_counter()

    # -----------------------------
    # Feature Engineering (same as training)
    # -----------------------------
    for col in ["l1d_size", "l1i_size", "l2_size",
                "l1d_assoc", "l1i_assoc", "l2_assoc"]:
        df[f"{col}_log2"] = numpy.log2(df[col])

    df["l2_l1d_ratio_log2"] = df["l2_size_log2"] - df["l1d_size_log2"]
    df["l1d_sets_log2"] = df["l1d_size_log2"] - df["l1d_assoc_log2"]
    df["l2_sets_log2"] = df["l2_size_log2"] - df["l2_assoc_log2"]

    # Output columns
    df["pred_ipc"] = numpy.nan
    df["pred_l2_miss_rate"] = numpy.nan
    df["warnings"] = ""

    # -----------------------------
    # Batch inference per workload
    # -----------------------------
    for workload, group_idx in df.groupby("workload").groups.items():

        if workload not in WORKLOADS:
            raise ValueError(f"Unknown workload: {workload}")
        
        X = df.loc[group_idx, FEATURES].values  # shape: (N, num_features)
        preds = {}

        for target in TARGETS:
            
            model, is_log_target = models[f"model_{workload}_{target}"]

            pred_raw = model.predict(X)
            pred = numpy.expm1(pred_raw) if is_log_target else pred_raw
            
            if target == "l2_miss_rate":
                pred = numpy.clip(pred, 0, 1)

            preds[target] = pred

        # Assign predictions
        df.loc[group_idx, "pred_ipc"] = preds["ipc"]
        df.loc[group_idx, "pred_l2_miss_rate"] = preds["l2_miss_rate"]

        # -----------------------------
        # Vectorized sanity checks
        # -----------------------------
        ipc_vals = preds["ipc"]
        miss_vals = preds["l2_miss_rate"]

        warnings = numpy.full(len(group_idx), "", dtype=object)

        bad_ipc = (ipc_vals < 0) | (ipc_vals > 3.5)
        bad_miss = (miss_vals < 0) | (miss_vals > 1)

        warnings[bad_ipc] = [f"IPC={v:.3f} out of physical range" for v in ipc_vals[bad_ipc]]
        warnings[bad_miss] = [
            (w + "; " if w else "") + f"L2 miss rate={v:.3f} out of [0,1]"
            for w, v in zip(warnings[bad_miss], miss_vals[bad_miss])
        ]

        df.loc[group_idx, "warnings"] = warnings

    elapsed = time.perf_counter() - start_time
    print(f"⏱️  Total processing time: {elapsed:.3f} sec")

    df.to_csv(output_path, index=False)
    print(f"✅ Inference complete. Results saved to: {output_path}")
    
    if (df["warnings"] != "").any():
        print("\n⚠️  Some rows triggered sanity warnings:")
        print(df[df["warnings"] != ""][["workload", "pred_ipc", "pred_l2_miss_rate", "warnings"]])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference_converted.py <input.csv> <output.csv>")
        sys.exit(1)

    models = load_models()
    run_inference(models, sys.argv[1], sys.argv[2])
