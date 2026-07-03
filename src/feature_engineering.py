import pandas as pd
import numpy as np

class FeatureEngineer:
    def credit_card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'Amount' in df_out.columns:
            df_out['amt_log'] = np.log1p(df_out['Amount'])
            df_out['amt_deviation'] = (df_out['Amount'] - df_out['Amount'].mean()) / df_out['Amount'].std()
            
        if 'Time' in df_out.columns:
            df_out['time_hour'] = (df_out['Time'] % 86400) // 3600
            df_out['is_night'] = df_out['time_hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
            
        df_out = df_out.drop(columns=['Amount', 'Time'], errors='ignore')
        return df_out

    def insurance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'deductible' in df_out.columns:
            df_out['high_deductible'] = df_out['deductible'].apply(lambda x: 1 if x >= 700 else 0)
        if 'number_of_past_complaints' in df_out.columns:
            df_out['multiple_claims'] = df_out['number_of_past_complaints'].apply(lambda x: 1 if x > 1 else 0)
        if 'months_as_customer' in df_out.columns:
            df_out['recent_claim'] = df_out['months_as_customer'].apply(lambda x: 1 if x < 6 else 0)
        if 'age' in df_out.columns:
            df_out['young_driver'] = df_out['age'].apply(lambda x: 1 if x <= 25 else 0)
            
        return df_out

    def ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'Shipping_Address' in df_out.columns and 'Billing_Address' in df_out.columns:
            df_out['address_mismatch'] = (df_out['Shipping_Address'] != df_out['Billing_Address']).astype(int)
        if 'Account_Age_Days' in df_out.columns:
            df_out['is_new_account'] = (df_out['Account_Age_Days'] < 30).astype(int)
        if 'Transaction_Amount' in df_out.columns:
            amt_95th = df_out['Transaction_Amount'].quantile(0.95)
            df_out['is_high_value'] = (df_out['Transaction_Amount'] > amt_95th).astype(int)
        if 'Transaction_Hour' in df_out.columns:
            df_out['is_unusual_hour'] = df_out['Transaction_Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
            
        return df_out

    def create_lstm_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        if n_samples <= seq_len:
            raise ValueError("Not enough samples to create sequences of this length.")
            
        X_seq = []
        y_seq = []
        
        for i in range(n_samples - seq_len + 1):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len - 1])
            
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
