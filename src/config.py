import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DIR        = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR  = os.path.join(DATA_DIR, "processed")
SAVED_MODELS   = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR    = os.path.join(RESULTS_DIR, "metrics")

# Raw file names 
CC_RAW_FILE    = os.path.join(RAW_DIR, "creditcard.csv")
INS_RAW_FILE   = os.path.join(RAW_DIR, "insurance_claims.csv")
ECO_RAW_FILE   = os.path.join(RAW_DIR, "ecommerce_fraud.csv")

# General hyperparameters
RANDOM_SEED        = 42
TEST_SIZE          = 0.15
VAL_SIZE           = 0.15
ECOMMERCE_SAMPLE   = 300000

# Autoencoder (TensorFlow/Keras) hyperparameters
AE_HIDDEN_DIMS     = [64, 32, 16]
AE_DROPOUT         = 0.2
AE_LR              = 0.001
AE_EPOCHS          = 50
AE_BATCH_SIZE      = 256
AE_PATIENCE        = 5
AE_THRESHOLD_PCT   = 95

# Insurance models hyperparameters
RF_N_ESTIMATORS    = 200
RF_MAX_DEPTH       = 10
RF_CV_FOLDS        = 5
XGB_LR             = 0.05
XGB_MAX_DEPTH      = 6
XGB_N_ESTIMATORS   = 200

# LSTM (TensorFlow/Keras) hyperparameters
LSTM_HIDDEN_UNITS  = 128
LSTM_LAYERS        = 2
LSTM_DROPOUT       = 0.3
LSTM_SEQ_LEN       = 10
LSTM_BATCH_SIZE    = 64
LSTM_EPOCHS        = 30
LSTM_LR            = 0.001
LSTM_PATIENCE      = 5

# Ensemble configuration
ENSEMBLE_WEIGHTS   = {"credit_card": 0.35, "insurance": 0.30, "ecommerce": 0.35}
FRAUD_THRESHOLD    = 0.5

# Create all necessary directories on import
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SAVED_MODELS, RESULTS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)
