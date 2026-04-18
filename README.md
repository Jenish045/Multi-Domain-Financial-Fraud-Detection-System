# 🛡️ Multi-Domain Financial Fraud Detection System

## 📌 Overview
This project implements a large-scale fraud detection system across three domains: credit card transactions, insurance claims, and e-commerce orders. The system processes **~1.75M+ records** and is designed to handle **extreme class imbalance (~0.17% fraud cases)** using domain-specific machine learning and deep learning models.

---

## 🎯 Objective
To design a robust fraud detection pipeline capable of identifying anomalous and fraudulent behavior across different financial systems while maintaining high recall and acceptable false positive rates.

---

## 🧠 Approach

- Credit Card Fraud → Autoencoder (unsupervised anomaly detection)  
- Insurance Fraud → XGBoost (supervised classification)  
- E-Commerce Fraud → LSTM (sequence-based deep learning)  

Each model is optimized independently for its domain.

---

## 📊 Model Performance

| Domain       | Model        | Precision | Recall  | F1 Score | ROC-AUC |
|--------------|-------------|----------|--------|---------|---------|
| Credit Card  | Autoencoder | 0.0283   | 0.8378 | 0.0547  | **0.9367** |
| Insurance    | XGBoost     | 0.6818   | 0.8108 | **0.7407** | **0.9033** |
| E-Commerce   | LSTM        | 0.2324   | 0.6948 | 0.3483  | **0.8378** |

---

## 🔍 Key Observations

- Autoencoder achieves **high ROC-AUC (0.9367)** but low precision due to extreme class imbalance  
- Insurance model provides the **best balance (F1: 0.7407)** between precision and recall  
- LSTM captures sequential fraud patterns with **moderate performance (AUC: 0.8378)**  
- High recall across models ensures most fraudulent cases are detected  

---

## ⚙️ Pipeline

Data Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment

---

## 📦 Dataset Summary

- Credit Card: 284,807 transactions  
- Insurance: 1,001 records  
- E-Commerce: 1,472,952 transactions  

Total: **~1.75M+ records**

---

## 🧠 Tech Stack

Python, Pandas, NumPy  
Scikit-learn, XGBoost  
TensorFlow / Keras  
Matplotlib, Seaborn, Plotly  
Streamlit  

---

## 📁 Project Structure

multi_domain_fraud_detection/
├── data/
├── src/
├── scripts/
├── dashboard/
├── saved_models/
├── results/

---

## ▶️ Run Instructions

pip install -r requirements.txt  

python scripts/train_autoencoder.py  
python scripts/train_random_forest.py  
python scripts/train_lstm.py  

streamlit run dashboard/app.py  

---

## 📈 Output

- Fraud probability scores  
- Model evaluation metrics  
- ROC curves and confusion matrices  
- Interactive dashboard  

---

## 🚀 Applications

- Financial fraud detection  
- Insurance claim validation  
- E-commerce transaction monitoring  
- Risk scoring systems  

---

## 📄 License

Academic / Educational Use