import os
import sys
import numpy as np

# Ensure this is executed from the project root or adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CC_RAW_FILE, AE_HIDDEN_DIMS, AE_DROPOUT, AE_LR, AE_EPOCHS, AE_BATCH_SIZE, AE_PATIENCE
from src.preprocessing import FraudPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.autoencoder import AutoencoderTrainer

def main():
    if not os.path.exists(CC_RAW_FILE):
        print(f"Error: Raw data file not found at {CC_RAW_FILE}.")
        print("Please download creditcard.csv from Kaggle and place it in the data/raw/ folder.")
        sys.exit(1)
        
    preprocessor = FraudPreprocessor()
    engineer = FeatureEngineer()
    
    print("Loading Credit Card Data...")
    df = preprocessor.load_credit_card(CC_RAW_FILE)
    
    print("Engineering features...")
    df_engineered = engineer.credit_card_features(df)
    
    print("Scaling and encoding...")
    y = df_engineered['Class'].values
    X_df = df_engineered.drop(columns=['Class'])
    
    # Encode (noop for CC) and Scale
    X_encoded = preprocessor.encode_features(X_df, 'credit_card')
    X_scaled = preprocessor.scale_features(X_encoded, 'credit_card', fit=True)
    X_scaled = X_scaled.astype(np.float32)
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X_scaled, y, temporal=False)
    
    # CRITICAL: Autoencoder trains on normal transactions only
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    
    trainer = AutoencoderTrainer(
        input_dim=X_train.shape[1],
        hidden_dims=AE_HIDDEN_DIMS,
        dropout_rate=AE_DROPOUT,
        lr=AE_LR,
        patience=AE_PATIENCE
    )
    
    print("Training Autoencoder...")
    trainer.train(X_train_normal, X_val, AE_EPOCHS, AE_BATCH_SIZE)
    
    print("Finding Anomaly Threshold...")
    threshold = trainer.find_threshold(X_val_normal)
    
    print(f"Evaluating on Test Set with Threshold: {threshold:.4f}...")
    metrics = trainer.evaluate(X_test, y_test, threshold)
    
    print("Training complete. Model saved to saved_models/")

if __name__ == "__main__":
    main()
