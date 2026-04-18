### Step 1 — Install Python
- Recommended: Python 3.10 or 3.11
- Download from: https://www.python.org/downloads/
- During install on Windows: CHECK "Add Python to PATH"
- Verify: open terminal and run: `python --version`

### Step 2 — Create project folder
  ```bash
  mkdir multi_domain_fraud_detection
  cd multi_domain_fraud_detection
  ```

### Step 3 — Create virtual environment
  ```bash
  python -m venv venv
  ```
  Activate it:
  - Windows:   `venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`

  You should see `(venv)` at the start of your terminal line.

### Step 4 — Install all dependencies
  ```bash
  pip install -r requirements.txt
  ```
  This will install:
  - TensorFlow 2.x (for Autoencoder + LSTM)
  - Scikit-learn (for Random Forest)
  - XGBoost
  - Streamlit (for dashboard)
  - Pandas, NumPy, Matplotlib, Seaborn, Plotly
  - imbalanced-learn (for SMOTE)
  - joblib (for saving sklearn models)

  Expected time: 5-10 minutes depending on internet speed.

### Step 5 — Download datasets from Kaggle

  You need a free Kaggle account. Go to https://www.kaggle.com

  **Dataset 1 — Credit Card Fraud:**
  - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - File to download: `creditcard.csv`
  - Place in: `data/raw/creditcard.csv`
  - Size: ~144 MB

  **Dataset 2 — Insurance Fraud:**
  - URL: https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
  - File to download: `insurance_claims.csv`
  - Place in: `data/raw/insurance_claims.csv`
  - Size: ~1.4 MB

  **Dataset 3 — E-Commerce Fraud:**
  - URL: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
  - File to download: `Fraudulent_E-Commerce_Transaction_Data.csv`
  - Place in: `data/raw/ecommerce_fraud.csv`
  - Size: ~200 MB

  **How to download from Kaggle:**
  1. Open the dataset URL
  2. Click the "Download" button (top right)
  3. Unzip the downloaded file
  4. Rename and move the CSV to `data/raw/` as shown above

### Step 6 — Verify folder looks like this before running scripts:
  ```text
  multi_domain_fraud_detection/
  ├── data/
  │   └── raw/
  │       ├── creditcard.csv
  │       ├── insurance_claims.csv
  │       └── ecommerce_fraud.csv
  ├── venv/
  └── requirements.txt
  ```

### Step 7 — Run training scripts IN ORDER:
  ```bash
  python scripts/train_autoencoder.py
  python scripts/train_random_forest.py
  python scripts/train_lstm.py
  python scripts/evaluate_all.py
  ```

### Step 8 — Launch dashboard:
  ```bash
  streamlit run dashboard/app.py
  ```
  This opens automatically in your browser at `http://localhost:8501`

### Common errors and fixes:
  - **Error:** `ModuleNotFoundError: No module named X`
    **Fix:** Make sure venv is activated, then run: `pip install X`
  - **Error:** `FileNotFoundError: data/raw/creditcard.csv not found`
    **Fix:** Check dataset is downloaded and placed in the correct path
  - **Error:** "OOM" or memory crash on e-commerce LSTM
    **Fix:** Open `src/config.py` and change `ECOMMERCE_SAMPLE = 100000`
  - **Error:** "Could not load model" in dashboard
    **Fix:** Run all 4 training scripts first before launching dashboard
