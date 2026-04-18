import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_CV_FOLDS, XGB_LR, 
                        XGB_MAX_DEPTH, XGB_N_ESTIMATORS, RANDOM_SEED, SAVED_MODELS, PLOTS_DIR, METRICS_DIR)

class InsuranceFraudDetector:
    """Class to test tree-based models for Insurance Fraud detection."""

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.best_model_name = None

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Cross-validates and trains a Random Forest.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
        Returns:
            RandomForestClassifier: Trained model.
        """
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight='balanced',
            random_state=RANDOM_SEED
        )
        
        skf = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_f1s = []
        cv_aucs = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val_fold)
            probs = model.predict_proba(X_val_fold)[:, 1]
            
            cv_f1s.append(f1_score(y_val_fold, preds, zero_division=0))
            cv_aucs.append(roc_auc_score(y_val_fold, probs))
            
        print(f"Random Forest CV Mean F1: {np.mean(cv_f1s):.4f}, Mean AUC: {np.mean(cv_aucs):.4f}")
        
        model.fit(X_train, y_train)
        self.rf_model = model
        joblib.dump(model, os.path.join(SAVED_MODELS, 'random_forest.pkl'))
        
        # Plot Feature Importance (assuming fewer features for Insurance)
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.title('Random Forest - Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'rf_importance.png'))
        plt.close()
        
        return model

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Trains an XGBoost model.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
        Returns:
            XGBClassifier: Trained model.
        """
        scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        
        model = XGBClassifier(
            learning_rate=XGB_LR,
            max_depth=XGB_MAX_DEPTH,
            n_estimators=XGB_N_ESTIMATORS,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            early_stopping_rounds=20,
            random_state=RANDOM_SEED
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.xgb_model = model
        joblib.dump(model, os.path.join(SAVED_MODELS, 'xgboost.pkl'))
        return model

    def compare_and_select(self, X_val: np.ndarray, y_val: np.ndarray) -> str:
        """Evaluates both models and selects the best one based on F1.
        
        Args:
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
        Returns:
            str: Name of the best model.
        """
        rf_preds = self.rf_model.predict(X_val)
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        
        xgb_preds = self.xgb_model.predict(X_val)
        xgb_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        
        rf_f1 = f1_score(y_val, rf_preds, zero_division=0)
        xgb_f1 = f1_score(y_val, xgb_preds, zero_division=0)
        
        print(f"{'Model':<15} | {'Precision':<9} | {'Recall':<6} | {'F1':<5} | {'AUC':<5}")
        print("-" * 50)
        print(f"{'Random Forest':<15} | {precision_score(y_val, rf_preds, zero_division=0):.4f}    | {recall_score(y_val, rf_preds, zero_division=0):.4f} | {rf_f1:.4f} | {roc_auc_score(y_val, rf_probs):.4f}")
        print(f"{'XGBoost':<15} | {precision_score(y_val, xgb_preds, zero_division=0):.4f}    | {recall_score(y_val, xgb_preds, zero_division=0):.4f} | {xgb_f1:.4f} | {roc_auc_score(y_val, xgb_probs):.4f}")
        
        if rf_f1 >= xgb_f1:
            self.best_model_name = "random_forest"
        else:
            self.best_model_name = "xgboost"
            
        with open(os.path.join(SAVED_MODELS, 'best_insurance_model.txt'), 'w') as f:
            f.write(self.best_model_name)
            
        return self.best_model_name

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts using the selected best model.
        
        Args:
            X (np.ndarray): Features.
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.best_model_name == "random_forest":
            model = self.rf_model if self.rf_model else joblib.load(os.path.join(SAVED_MODELS, 'random_forest.pkl'))
        else:
            model = self.xgb_model if self.xgb_model else joblib.load(os.path.join(SAVED_MODELS, 'xgboost.pkl'))
            
        return model.predict(X), model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Fully evaluates the chosen model.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
        Returns:
            dict: Evaluated metrics.
        """
        preds, probs = self.predict(X_test)
        
        metrics = {
            'precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'f1': round(f1_score(y_test, preds, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, probs), 4)
        }
        
        print(f"\nInsurance Model Evaluation ({self.best_model_name}):")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Insurance')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'cm_insurance.png'))
        plt.close()
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve - Insurance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roc_insurance.png'))
        plt.close()
        
        with open(os.path.join(METRICS_DIR, 'insurance_metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        return metrics
