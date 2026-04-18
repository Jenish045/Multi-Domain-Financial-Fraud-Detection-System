# 🛡️ Multi-Domain Financial Fraud Detection System

A complete end-to-end machine learning system for detecting fraudulent activities across multiple domains: **Credit Card Transactions, Insurance Claims, and E-Commerce Orders**. The system combines classical machine learning and deep learning models with a unified ensemble approach and an interactive dashboard for real-time fraud analysis.

---

## 🚀 Overview

This project is designed to solve real-world fraud detection problems across different domains using domain-specific modeling techniques. It processes **~1.75M+ transactions** and addresses **extreme class imbalance (~0.17% fraud cases)**.

Each domain is handled independently using the most suitable modeling approach, and results are combined using an ensemble layer to improve overall robustness.

---

## 🧠 Model Architecture

The system consists of three independent models:

- **Credit Card Fraud Detection**
  - Model: Autoencoder (Unsupervised)
  - Goal: Detect anomalies using reconstruction error
  - Strength: Works well for highly imbalanced datasets

- **Insurance Fraud Detection**
  - Models: Random Forest, XGBoost
  - Goal: Supervised classification of fraudulent claims
  - Best Model: XGBoost (based on validation performance)

- **E-Commerce Fraud Detection**
  - Model: LSTM (Deep Learning)
  - Goal: Capture sequential patterns in transaction behavior

- **Ensemble Layer**
  - Combines outputs using weighted averaging
  - Improves prediction consistency across domains

---

## ⚙️ Key Features

- Multi-domain fraud detection system (3 domains + ensemble)
- Handles extreme class imbalance using **SMOTE**
- End-to-end pipeline:
  Preprocessing → Feature Engineering → Training → Evaluation → Deployment
- Modular and scalable architecture
- Real-time fraud prediction using **Streamlit dashboard**
- Config-driven design for reproducibility
- Model persistence and reuse

---

## 📊 Model Performance

| Domain        | Model        | Precision | Recall  | F1 Score | ROC-AUC |
|---------------|------------- |---------- |-------- |--------- |---------|
| Credit Card   | Autoencoder  | 0.0283    | 0.8378  | 0.0547   | **0.9367** |
| Insurance     | XGBoost      | 0.6818    | 0.8108  | **0.7407** | **0.9033** |
| E-Commerce    | LSTM         | 0.2324    | 0.6948  | 0.3483   | **0.8378** |

---

## 📦 Datasets

The system uses three real-world datasets:

- **Credit Card Dataset**
  - ~284,807 transactions
  - Highly imbalanced (~0.17% fraud)

- **Insurance Claims Dataset**
  - ~1,000 records
  - Structured tabular data

- **E-Commerce Dataset**
  - ~1,472,952 transactions
  - Behavioral + transactional features

Total processed: **~1.75M+ records**

---

## 🧠 Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Deep Learning:** TensorFlow / Keras  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Frontend/UI:** Streamlit  

---

## 📁 Project Structure

multi_domain_fraud_detection/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── models/
│       ├── autoencoder.py
│       ├── random_forest.py
│       ├── lstm_model.py
│       └── ensemble.py
├── scripts/
│   ├── train_autoencoder.py
│   ├── train_random_forest.py
│   ├── train_lstm.py
│   └── evaluate_all.py
├── dashboard/
│   └── app.py
├── saved_models/
├── results/
├── requirements.txt
└── README.md

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train models
python scripts/train_autoencoder.py  
python scripts/train_random_forest.py  
python scripts/train_lstm.py  

### 3. Evaluate all models
python scripts/evaluate_all.py  

### 4. Launch dashboard
streamlit run dashboard/app.py  

---

## 📈 Outputs

- Trained models saved in `saved_models/`
- Metrics stored in `results/metrics/`
- Visualizations (ROC curves, confusion matrices) in `results/plots/`
- Interactive dashboard for real-time predictions

---

## 📌 Applications

- Financial transaction monitoring  
- Insurance fraud detection  
- E-commerce fraud prevention  
- Real-time risk scoring systems  

---

## 🔥 Highlights

- Processes **~1.75M+ transactions across domains**
- Combines **unsupervised + supervised + deep learning models**
- Handles **extreme class imbalance**
- Designed for **real-world scalability and deployment**

---

## 📄 License

This project is for academic and educational purposes.