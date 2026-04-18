import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INS_RAW_FILE
from src.preprocessing import FraudPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.random_forest import InsuranceFraudDetector

def main():
    if not os.path.exists(INS_RAW_FILE):
        print(f"Error: Raw data file not found at {INS_RAW_FILE}.")
        print("Please download insurance_claims.csv from Kaggle and place it in the data/raw/ folder.")
        sys.exit(1)

    preprocessor = FraudPreprocessor()
    engineer = FeatureEngineer()
    detector = InsuranceFraudDetector()

    print("Loading Insurance Data...")
    df = preprocessor.load_insurance(INS_RAW_FILE)

    print("Engineering features...")
    df_engineered = engineer.insurance_features(df)

    print("Encoding and splitting...")
    y = df_engineered['fraud_reported'].values
    X_df = df_engineered.drop(columns=['fraud_reported'])

    X_encoded = preprocessor.encode_features(X_df, 'insurance')
    
    # Split first to avoid data leakage
    X_temp, X_test, y_temp, y_test = preprocessor.split_data(X_encoded.values, y, temporal=False)[:4] # only want train/test splits for now, we'll do proper below
    # Let's just use the robust split data from preprocessor
    X_train_unscaled, X_val_unscaled, X_test_unscaled, y_train, y_val, y_test = preprocessor.split_data(X_encoded.values, y, temporal=False)

    print("Scaling features...")
    X_train_scaled = preprocessor.scale_features(X_train_unscaled, 'insurance', fit=True)
    X_val_scaled = preprocessor.scale_features(X_val_unscaled, 'insurance', fit=False)
    X_test_scaled = preprocessor.scale_features(X_test_unscaled, 'insurance', fit=False)

    print("Applying SMOTE to training data...")
    X_train_smote, y_train_smote = preprocessor.apply_smote(X_train_scaled, y_train)

    print("\nTraining Random Forest...")
    detector.train_random_forest(X_train_smote, y_train_smote)

    print("\nTraining XGBoost...")
    detector.train_xgboost(X_train_smote, y_train_smote, X_val_scaled, y_val)

    print("\nComparing models on Validation Set...")
    best_model = detector.compare_and_select(X_val_scaled, y_val)
    print(f"\nBest model selected: {best_model}")

    print("\nEvaluating Best Model on Test Set...")
    detector.evaluate(X_test_scaled, y_test)

    print("\nTraining complete. Model saved to saved_models/")

if __name__ == "__main__":
    main()
