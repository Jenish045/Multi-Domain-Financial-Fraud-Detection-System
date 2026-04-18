import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from src.config import FRAUD_THRESHOLD, SAVED_MODELS, PLOTS_DIR, METRICS_DIR, RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)

def build_lstm(seq_len: int, n_features: int, hidden_units: int, dropout_rate: float, lr: float) -> keras.Model:
    """Builds LSTM model using Keras Sequential API.
    
    Args:
        seq_len (int): Length of sequence.
        n_features (int): Number of features.
        hidden_units (int): Number of hidden units in LSTM.
        dropout_rate (float): Dropout rate.
        lr (float): Learning rate.
    Returns:
        keras.Model: Compiled model.
    """
    model = keras.Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(hidden_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(hidden_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='binary_crossentropy',
                  metrics=['accuracy', AUC(name='auc')])
    return model

class LSTMTrainer:
    """Trainer class for LSTM Sequence model for Ecommerce data."""
    
    def __init__(self, seq_len: int, n_features: int, hidden_units: int, dropout_rate: float, lr: float, patience: int):
        self.model = build_lstm(seq_len, n_features, hidden_units, dropout_rate, lr)
        self.patience = patience
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int, batch_size: int):
        """Trains the LSTM model with early stopping.
        
        Args:
            X_train (np.ndarray): Training inputs.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation inputs.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
        Returns:
            keras.callbacks.History: History object.
        """
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=self.patience, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        unique_classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
        class_weight = {int(k): v for k, v in zip(unique_classes, weights)}
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save(os.path.join(SAVED_MODELS, 'lstm_model.keras'))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(history.history['loss'], label='Train')
        ax1.plot(history.history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        ax2.plot(history.history['auc'], label='Train')
        ax2.plot(history.history['val_auc'], label='Validation')
        ax2.set_title('AUC')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'lstm_training.png'))
        plt.close()
        
        return history

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts anomalies based on the threshold.
        
        Args:
            X (np.ndarray): Sequence features.
        Returns:
            tuple: (fraud_labels, fraud_probabilities)
        """
        fraud_probs = self.model.predict(X, verbose=0).flatten()
        fraud_labels = (fraud_probs >= FRAUD_THRESHOLD).astype(int)
        return fraud_labels, fraud_probs

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluates LSTM model.
        
        Args:
            X_test (np.ndarray): Test sequences.
            y_test (np.ndarray): Test labels.
        Returns:
            dict: Evaluation metrics.
        """
        preds, probs = self.predict(X_test)
        
        metrics = {
            'precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'f1': round(f1_score(y_test, preds, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, probs), 4)
        }
        
        print(f"\nLSTM Model Evaluation:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - E-Commerce (LSTM)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'cm_ecommerce.png'))
        plt.close()
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve - E-Commerce (LSTM)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roc_ecommerce.png'))
        plt.close()
        
        with open(os.path.join(METRICS_DIR, 'ecommerce_metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        return metrics
