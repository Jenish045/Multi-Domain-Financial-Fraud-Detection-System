from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Jenish045/multi-domain-financial-fraud-detection-models"

MODEL_FILES = [
    "ae_threshold.pkl",
    "autoencoder.keras",
    "best_insurance_model.txt",
    "ecommerce_columns.pkl",
    "ecommerce_label_encoders.pkl",
    "ecommerce_medians.pkl",
    "insurance_columns.pkl",
    "insurance_label_encoders.pkl",
    "insurance_medians.pkl",
    "lstm_model.keras",
    "random_forest.pkl",
    "scaler_credit_card.pkl",
    "scaler_ecommerce.pkl",
    "scaler_insurance.pkl",
    "xgboost.pkl"
]

def download_models(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for file in MODEL_FILES:
        destination = save_dir / file

        if destination.exists():
            continue

        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            local_dir=save_dir
        )