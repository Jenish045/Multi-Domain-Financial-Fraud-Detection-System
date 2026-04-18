import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.preprocessing import FraudPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMTrainer

def main():
    if not os.path.exists(ECO_RAW_FILE):
        print(f"Error: Raw data file not found at {ECO_RAW_FILE}")
        sys.exit(1)

    preprocessor = FraudPreprocessor()
    engineer = FeatureEngineer()

    print("Loading E-Commerce Data...")
    df = preprocessor.load_ecommerce(ECO_RAW_FILE, ECOMMERCE_SAMPLE)

    print("Engineering features...")
    df_engineered = engineer.ecommerce_features(df)

    print("Encoding features...")
    y = df_engineered['Is_Fraudulent'].values
    X_df = df_engineered.drop(columns=['Is_Fraudulent'])

    X_encoded = preprocessor.encode_features(X_df, 'ecommerce')

    # 🔍 Debug check (IMPORTANT)
    print("Data types after encoding:\n", X_encoded.dtypes)

    # Ensure numeric
    X_encoded = X_encoded.astype(float)

    print("Scaling features...")
    X_scaled = preprocessor.scale_features(X_encoded.values, 'ecommerce', fit=True)
    X_scaled = X_scaled.astype(np.float32)

    print("Creating LSTM sequences...")
    X_seq, y_seq = engineer.create_lstm_sequences(X_scaled, y, LSTM_SEQ_LEN)

    print(f"Splitting temporal data... Total sequences: {len(X_seq)}")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_seq, y_seq, temporal=True
    )

    trainer = LSTMTrainer(
        seq_len=LSTM_SEQ_LEN,
        n_features=X_train.shape[2],
        hidden_units=LSTM_HIDDEN_UNITS,
        dropout_rate=LSTM_DROPOUT,
        lr=LSTM_LR,
        patience=LSTM_PATIENCE
    )

    print("Training LSTM Model...")
    trainer.train(X_train, y_train, X_val, y_val, LSTM_EPOCHS, LSTM_BATCH_SIZE)

    print("Evaluating on Test Set...")
    trainer.evaluate(X_test, y_test)

    print("\nTraining complete.")

if __name__ == "__main__":
    main()