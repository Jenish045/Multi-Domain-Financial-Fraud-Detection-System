import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
from src.config import AE_THRESHOLD_PCT, SAVED_MODELS, PLOTS_DIR, METRICS_DIR, RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)

def build_autoencoder(input_dim: int, hidden_dims: list, dropout_rate: float) -> keras.Model:
    """Builds autoencoder model using Keras Functional API.
    
    Args:
        input_dim (int): Number of input features.
        hidden_dims (list): List of integers for hidden layer sizes.
        dropout_rate (float): Dropout probability.
    Returns:
        keras.Model: Compiled autoencoder model.
    """
    inputs = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(hidden_dims[0], activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_dims[1], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    encoded = Dense(hidden_dims[2], activation='relu')(x)
    
    # Decoder
    x = Dense(hidden_dims[1], activation='relu')(encoded)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_dims[0], activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    decoded = Dense(input_dim, activation='linear')(x)
    
    model = keras.Model(inputs, decoded)
    return model

class AutoencoderTrainer:
    """Trainer for the unsupervised autoencoder model."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float, lr: float, patience: int):
        self.model = build_autoencoder(input_dim, hidden_dims, dropout_rate)
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.patience = patience
        self.threshold = None
        
    def train(self, X_train_normal: np.ndarray, X_val: np.ndarray, epochs: int, batch_size: int):
        """Trains the autoencoder on non-fraud samples only.
        
        Args:
            X_train_normal (np.ndarray): Normal data for training.
            X_val (np.ndarray): Validation data.
            epochs (int): Max epochs.
            batch_size (int): Batch size.
        Returns:
            keras.callbacks.History: Training history object.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save(os.path.join(SAVED_MODELS, 'autoencoder.keras'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'autoencoder_loss.png'))
        plt.close()
        
        return history
        
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Calculates MSE between input and reconstruction.
        
        Args:
            X (np.ndarray): Input features.
        Returns:
            np.ndarray: Reconstruction errors per sample.
        """
        X_pred = self.model.predict(X, verbose=0)
        errors = np.mean(np.power(X - X_pred, 2), axis=1)
        return errors
        
    def find_threshold(self, X_val_normal: np.ndarray) -> float:
        """Determines the anomaly detection threshold.
        
        Args:
            X_val_normal (np.ndarray): Normal validation data.
        Returns:
            float: Threshold value.
        """
        errors = self.get_reconstruction_errors(X_val_normal)
        self.threshold = np.percentile(errors, AE_THRESHOLD_PCT)
        joblib.dump(self.threshold, os.path.join(SAVED_MODELS, 'ae_threshold.pkl'))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50, kde=True)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold:.4f})')
        plt.title('Reconstruction Error Distribution (Normal Data)')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ae_threshold.png'))
        plt.close()
        
        return self.threshold
        
    def predict(self, X: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Predicts anomalous and computes pseudo-probabilities.
        
        Args:
            X (np.ndarray): Input features.
            threshold (float): Anomaly threshold.
        Returns:
            tuple: (fraud_labels, fraud_probs)
        """
        errors = self.get_reconstruction_errors(X)
        fraud_labels = (errors > threshold).astype(int)
        
        # Min-max scale errors pseudo-probabilities (cap dynamically based on distribution)
        # Using a simple heuristic for pseudo-probability.
        min_err = np.min(errors)
        max_err = np.percentile(errors, 99) # limit outliers effect
        probs = np.clip((errors - min_err) / (max_err - min_err + 1e-8), 0, 1)
        
        return fraud_labels, probs
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float) -> dict:
        """Evaluates autoencoder on test set.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            threshold (float): Used threshold.
        Returns:
            dict: Dictionary of metrics.
        """
        preds, probs = self.predict(X_test, threshold)
        
        metrics = {
            'precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'f1': round(f1_score(y_test, preds, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, probs), 4)
        }
        
        print(f"\nAutoencoder Evaluation:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Autoencoder')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'cm_autoencoder.png'))
        plt.close()
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve - Autoencoder')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roc_autoencoder.png'))
        plt.close()
        
        with open(os.path.join(METRICS_DIR, 'autoencoder_metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        return metrics
