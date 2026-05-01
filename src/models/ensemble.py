from dataclasses import dataclass
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from src.config import SAVED_MODELS, LSTM_SEQ_LEN

@dataclass
class FraudResult:
    domain: str
    fraud_probability: float
    fraud_label: bool
    alert_level: str
    model_used: str
    confidence: str

def get_alert_level(prob: float) -> str:
    """Gets alert level based on probability."""
    if prob < 0.4: return "Low"
    elif prob < 0.7: return "Medium"
    else: return "High"

def get_confidence(prob: float) -> str:
    """Gets confidence score based on margin from decision boundary."""
    dist_from_05 = abs(prob - 0.5)
    if dist_from_05 > 0.35: return "High"
    elif dist_from_05 > 0.15: return "Medium"
    else: return "Low"

class FraudEnsemble:
    """Ensemble for predicting fraud probabilities across all three domains."""

    def __init__(self):
        self.models_loaded = False
        self.ae_model = None
        self.ae_threshold = None
        self.rf_model = None
        self.lstm_model = None
        
        self.scaler_cc = None
        self.scaler_ins = None
        self.scaler_eco = None
        
        # Insurance inference artifacts
        self.ins_label_encoders = None
        self.ins_columns = None
        self.ins_medians = None
        
        # E-commerce inference artifacts
        self.eco_label_encoders = None
        self.eco_columns = None
        self.eco_medians = None
        
    def load_models(self):
        """Loads all required models and scalers."""
        self.ae_model = keras.models.load_model(os.path.join(SAVED_MODELS, 'autoencoder.keras'))
        self.ae_threshold = joblib.load(os.path.join(SAVED_MODELS, 'ae_threshold.pkl'))
        
        with open(os.path.join(SAVED_MODELS, 'best_insurance_model.txt'), 'r') as f:
            best_ins = f.read().strip()
        self.rf_model = joblib.load(os.path.join(SAVED_MODELS, f"{best_ins}.pkl"))
        
        self.lstm_model = keras.models.load_model(os.path.join(SAVED_MODELS, 'lstm_model.keras'))
        
        self.scaler_cc = joblib.load(os.path.join(SAVED_MODELS, 'scaler_credit_card.pkl'))
        self.scaler_ins = joblib.load(os.path.join(SAVED_MODELS, 'scaler_insurance.pkl'))
        self.scaler_eco = joblib.load(os.path.join(SAVED_MODELS, 'scaler_ecommerce.pkl'))
        
        # Load insurance inference artifacts
        self.ins_label_encoders = joblib.load(os.path.join(SAVED_MODELS, 'insurance_label_encoders.pkl'))
        self.ins_columns = joblib.load(os.path.join(SAVED_MODELS, 'insurance_columns.pkl'))
        self.ins_medians = joblib.load(os.path.join(SAVED_MODELS, 'insurance_medians.pkl'))
        
        # Load e-commerce inference artifacts
        self.eco_label_encoders = joblib.load(os.path.join(SAVED_MODELS, 'ecommerce_label_encoders.pkl'))
        self.eco_columns = joblib.load(os.path.join(SAVED_MODELS, 'ecommerce_columns.pkl'))
        self.eco_medians = joblib.load(os.path.join(SAVED_MODELS, 'ecommerce_medians.pkl'))
        
        self.models_loaded = True
        print("All models loaded successfully")

    def predict_credit_card(self, input_dict: dict) -> FraudResult:
        """Predicts credit card fraud instance.
        
        Args:
            input_dict (dict): Dictionary with features.
        Returns:
            FraudResult: Result wrapper object.
        """
        cols = [f'V{i}' for i in range(1, 29)] + ['amt_log', 'amt_deviation', 'time_hour', 'is_night']
        df = pd.DataFrame([input_dict])
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0 # simple default for missing synthetic features
        df = df[cols]
        
        X_scaled = self.scaler_cc.transform(df)
        X_scaled = X_scaled.astype(np.float32)
        
        errors = np.mean(np.power(X_scaled - self.ae_model.predict(X_scaled, verbose=0), 2), axis=1)
        prob = np.clip((errors[0] - np.min(errors)) / (np.percentile(errors, 99) - np.min(errors) + 1e-8), 0, 1)
        # Using simple threshold rule as placeholder for dynamically scaled prob.
        # Alternatively, we could just say anything above threshold has >0.5 probability (cap it relative to threshold).
        # A simpler way assuming error ranges are similar:
        is_fraud = bool(errors[0] > self.ae_threshold)
        final_prob = min(errors[0] / (self.ae_threshold * 2), 1.0) if not is_fraud else max(0.51, min(errors[0] / (self.ae_threshold * 2), 1.0))
        
        return FraudResult(
            domain="Credit Card",
            fraud_probability=float(final_prob),
            fraud_label=is_fraud,
            alert_level=get_alert_level(final_prob),
            model_used="Autoencoder",
            confidence=get_confidence(final_prob)
        )

    def predict_insurance(self, input_dict: dict) -> FraudResult:
        """Predicts insurance fraud instance.
        
        Args:
            input_dict (dict): Dictionary of features from the dashboard form.
        Returns:
            FraudResult: Result wrapper object.
        """
        import warnings
        
        # Start with median defaults for all 37 training columns
        row = dict(self.ins_medians)
        
        # Map dashboard form fields to their corresponding training columns
        field_mapping = {
            'age': 'age',
            'months_as_customer': 'months_as_customer',
            'deductible': 'policy_deductable',
            'total_claim_amount': 'total_claim_amount',
            'injury_claim': 'injury_claim',
            'property_claim': 'property_claim',
            'vehicle_claim': 'vehicle_claim',
            'policy_annual_premium': 'policy_annual_premium',
            'bodily_injuries': 'bodily_injuries',
            'witnesses': 'witnesses',
            'number_of_vehicles_involved': 'number_of_vehicles_involved',
            'incident_hour_of_the_day': 'incident_hour_of_the_day',
            'recent_claim': 'recent_claim',
            'young_driver': 'young_driver',
        }
        
        # Overwrite numeric fields from form input
        for form_key, col_name in field_mapping.items():
            if form_key in input_dict and col_name in row:
                row[col_name] = input_dict[form_key]
        
        # Encode categorical fields from the form using saved label encoders
        categorical_mapping = {
            'incident_type': 'incident_type',
            'collision_type': 'collision_type',
            'incident_severity': 'incident_severity',
            'authorities_contacted': 'authorities_contacted',
            'property_damage': 'property_damage',
            'police_report_available': 'police_report_available',
        }
        for form_key, col_name in categorical_mapping.items():
            if form_key in input_dict and col_name in self.ins_label_encoders:
                le = self.ins_label_encoders[col_name]
                value = input_dict[form_key]
                if value in le.classes_:
                    row[col_name] = le.transform([value])[0]
                # else: keep the median default (already numeric)
        
        # Build DataFrame in the exact column order used during training
        df = pd.DataFrame([row])[self.ins_columns]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = self.scaler_ins.transform(df)
        
        X_scaled = X_scaled.astype(np.float32)
        prob = float(self.rf_model.predict_proba(X_scaled)[:, 1][0])
        
        return FraudResult(
            domain="Insurance",
            fraud_probability=prob,
            fraud_label=bool(prob > 0.5),
            alert_level=get_alert_level(prob),
            model_used="Random Forest / XGBoost",
            confidence=get_confidence(prob)
        )

    def predict_ecommerce(self, input_dict: dict) -> FraudResult:
        """Predicts ecommerce fraud instance.
        
        Args:
            input_dict (dict): Dictionary of features from the dashboard form.
        Returns:
            FraudResult: Result wrapper object.
        """
        import warnings
        
        # Start with median defaults for all training columns
        row = dict(self.eco_medians)
        
        # Map dashboard form fields to their corresponding training columns
        field_mapping = {
            'TransactionAmount': 'Transaction_Amount',
            'AccountAgeDays': 'Account_Age_Days',
            'TransactionHour': 'Transaction_Hour',
            'is_new_account': 'is_new_account' if 'is_new_account' in row else None,
            'address_mismatch': 'address_mismatch' if 'address_mismatch' in row else None,
            'is_high_value': 'is_high_value' if 'is_high_value' in row else None,
            'is_unusual_hour': 'is_unusual_hour' if 'is_unusual_hour' in row else None,
        }
        
        # Overwrite fields from form input
        for form_key, col_name in field_mapping.items():
            if col_name and form_key in input_dict and col_name in row:
                row[col_name] = input_dict[form_key]
        
        # Also set derived time fields from TransactionHour
        if 'TransactionHour' in input_dict:
            hr = input_dict['TransactionHour']
            if 'hour' in row:
                row['hour'] = hr
            if 'is_weekend' in row and 'is_weekend' not in input_dict:
                row['is_weekend'] = 0  # default to weekday
        
        # Build DataFrame in the exact column order used during training
        df = pd.DataFrame([row])[self.eco_columns]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = self.scaler_eco.transform(df)
        X_scaled = X_scaled.astype(np.float32)
        
        # Reshape for LSTM (pad with zeros for multiple timesteps)
        n_features = X_scaled.shape[1]
        seq = np.zeros((1, LSTM_SEQ_LEN, n_features), dtype=np.float32)
        seq[0, -1, :] = X_scaled[0]
        
        prob = float(self.lstm_model.predict(seq, verbose=0).flatten()[0])
        
        return FraudResult(
            domain="E-Commerce",
            fraud_probability=prob,
            fraud_label=bool(prob > 0.5),
            alert_level=get_alert_level(prob),
            model_used="LSTM",
            confidence=get_confidence(prob)
        )
