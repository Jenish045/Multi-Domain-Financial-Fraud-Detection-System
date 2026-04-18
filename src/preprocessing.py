import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_SEED, SAVED_MODELS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FraudPreprocessor:
    """Class to handle preprocessing logic for all three datasets."""

    def load_credit_card(self, filepath: str) -> pd.DataFrame:
        """Loads credit card dataset and performs basic checks.
        
        Args:
            filepath (str): Path to creditcard.csv.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        logging.info(f"Credit Card shape: {df.shape}")
        logging.info(f"Credit Card class distribution:\n{df['Class'].value_counts()}")
        
        assert not df.isnull().values.any(), "Missing values found in credit card dataset!"
        return df

    def load_insurance(self, filepath: str) -> pd.DataFrame:
        """Loads insurance dataset and handles missing values.
        
        Args:
            filepath (str): Path to insurance_claims.csv.
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
            
        df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
        
        cols_to_drop = ['policy_number', 'policy_bind_date', 'insured_zip', '_c39']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        logging.info(f"Insurance shape: {df.shape}")
        logging.info(f"Insurance class distribution:\n{df['fraud_reported'].value_counts()}")
        return df

    def load_ecommerce(self, filepath: str, sample_size: int) -> pd.DataFrame:
        from sklearn.model_selection import train_test_split

        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.replace(' ', '_')
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        # Stratified sampling (correct way)
        if sample_size < len(df):
            df, _ = train_test_split(
                df,
                train_size=sample_size / len(df),  # convert to fraction
                stratify=df['Is_Fraudulent'],
                random_state=RANDOM_SEED
            )
            df = df.reset_index(drop=True)

        # Date feature engineering
        if 'Transaction_Date' in df.columns:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
            df['hour'] = df['Transaction_Date'].dt.hour
            df['day_of_week'] = df['Transaction_Date'].dt.dayofweek
            df['month'] = df['Transaction_Date'].dt.month
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df = df.drop(columns=['Transaction_Date'])

        # Handle missing values
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        logging.info(f"E-commerce shape: {df.shape}")
        logging.info(f"E-commerce class distribution:\n{df['Is_Fraudulent'].value_counts()}")

        return df

    def encode_features(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Encodes features based on the domain.
        
        Args:
            df (pd.DataFrame): Data.
            domain (str): Domain ('credit_card', 'insurance', 'ecommerce').
        Returns:
            pd.DataFrame: Data with encoded features.
        """
        df_encoded = df.copy()
        if domain == 'credit_card':
            pass # V1-V28 are already numeric, Time and Amount will be scaled later
        elif domain == 'insurance':
            le = LabelEncoder()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        elif domain == 'ecommerce':
            df_encoded = df_encoded.copy()

            # 🔥 Drop ID-like columns (high cardinality → useless + breaks scaling)
            drop_cols = []
            for col in df_encoded.columns:
                if df_encoded[col].dtype == 'object' and df_encoded[col].nunique() > 0.9 * len(df_encoded):
                    drop_cols.append(col)

            df_encoded = df_encoded.drop(columns=drop_cols)

            # 🔥 Encode ALL remaining categorical columns
            le = LabelEncoder()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                
        return df_encoded

    def scale_features(self, X: pd.DataFrame, domain: str, fit: bool = True) -> np.ndarray:
        """Scales numeric features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Features.
            domain (str): Domain tag for saving the scaler.
            fit (bool): Whether to fit or just load and transform.
        Returns:
            np.ndarray: Scaled features array.
        """
        scaler_path = os.path.join(SAVED_MODELS, f"scaler_{domain}.pkl")
        
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)
        else:
            try:
                scaler = joblib.load(scaler_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            X_scaled = scaler.transform(X)
            
        return X_scaled

    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies SMOTE to balance the training data classes.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
        Returns:
            tuple: Resampled (X, y).
        """
        unique, counts = np.unique(y_train, return_counts=True)
        logging.info(f"Class distribution before SMOTE: {dict(zip(unique, counts))}")
        
        smote = SMOTE(random_state=RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        unique_res, counts_res = np.unique(y_res, return_counts=True)
        logging.info(f"Class distribution after SMOTE: {dict(zip(unique_res, counts_res))}")
        
        return X_res, y_res

    def split_data(self, X: np.ndarray, y: np.ndarray, temporal: bool = False) -> tuple:
        """Splits data into train, val, test sets.
        
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Targets.
            temporal (bool): If True, split sequentially instead of random stratify.
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        from src.config import TEST_SIZE, VAL_SIZE
        
        if temporal:
            n = len(X)
            idx1 = int(n * (1.0 - TEST_SIZE - VAL_SIZE))
            idx2 = int(n * (1.0 - TEST_SIZE))
            
            X_train, y_train = X[:idx1], y[:idx1]
            X_val, y_val = X[idx1:idx2], y[idx1:idx2]
            X_test, y_test = X[idx2:], y[idx2:]
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
            )
            val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
            )
            
        return X_train, X_val, X_test, y_train, y_val, y_test
