# Script used to convert the trained model's pickle files to XGBoost's natively supported JSON format

import os, json, joblib, xgboost

IN_MODELS_DIR = "surrogate_models_v2"
OUT_MODELS_DIR = "surrogate_models_v2_json"

os.makedirs(OUT_MODELS_DIR, exist_ok=True)

for file in os.listdir(IN_MODELS_DIR):
    if not file.endswith('.pkl'):
        continue

    path = os.path.join(IN_MODELS_DIR, file)
    pickle = joblib.load(path)

    features: list = pickle['features']
    log_target: bool = pickle['log_target']
    model: xgboost.XGBRegressor = pickle['model']

    print(f"Converting pickle: {pickle}")
    base_name = file.replace('.pkl', '')
    
    model_output = os.path.join(OUT_MODELS_DIR, f"{base_name}.json")
    model.save_model(model_output)

    metadata_output = os.path.join(OUT_MODELS_DIR, f"{base_name}_meta.json")
    with open(metadata_output, "w") as file:
        json.dump({
                "features": features,
                "log_target": log_target
            }, file, indent=4)
        
    print(f"Converted {file}")