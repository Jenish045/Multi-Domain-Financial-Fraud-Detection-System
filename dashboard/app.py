import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ensemble import FraudEnsemble
from src.config import PLOTS_DIR

st.set_page_config(page_title="FraudGuard AI", layout="wide", page_icon="🛡")

@st.cache_resource
def load_all_models():
    ensemble = FraudEnsemble()
    try:
        ensemble.load_models()
        return ensemble, True
    except Exception as e:
        return None, False

ensemble, models_loaded = load_all_models()

if not models_loaded:
    st.error("Models not found. Please run all training scripts first.")
    st.code("python scripts/train_autoencoder.py")
    st.code("python scripts/train_random_forest.py")
    st.code("python scripts/train_lstm.py")
    st.stop()

def show_fraud_gauge(prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 40], "color": "#EAF3DE"},
                {"range": [40, 70], "color": "#FAEEDA"},
                {"range": [70, 100], "color": "#FCEBEB"}
            ],
            "threshold": {"value": round(prob * 100, 1), "line": {"color": "red", "width": 4}}
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

def show_alert_badge(alert_level: str):
    colors = {"Low": "#27500A", "Medium": "#633806", "High": "#791F1F"}
    bgs = {"Low": "#EAF3DE", "Medium": "#FAEEDA", "High": "#FCEBEB"}
    
    color = colors.get(alert_level, "#000")
    bg = bgs.get(alert_level, "#fff")
    
    st.markdown(
        f'<div style="background-color: {bg}; color: {color}; padding: 10px; '
        f'border-radius: 5px; text-align: center; font-weight: bold; '
        f'margin-bottom: 20px;">{alert_level.upper()} RISK</div>',
        unsafe_allow_html=True
    )

def show_result_metrics(result):
    st.info(f"Model used: **{result.model_used}** | Confidence: **{result.confidence}**")

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("FraudGuard AI")
st.sidebar.markdown("Multi-Domain Fraud Detection")
st.sidebar.divider()
st.sidebar.markdown("""
**Active Models:**
- 💳 **Credit Card** → Autoencoder (TensorFlow)
- 🚗 **Insurance**   → Random Forest / XGBoost
- 🛒 **E-Commerce**  → LSTM (TensorFlow)
""")

tabs = st.tabs(["Credit Card", "Insurance", "E-Commerce", "Analytics"])

with tabs[0]:
    st.header("Credit Card Transaction Checker")
    with st.form("cc_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=150.0)
            transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
        with col2:
            is_night = st.checkbox("Is Night Transaction?")
            amt_deviation = st.number_input("Amount Deviation From Mean", value=0.0)
        submitted = st.form_submit_button("Check Transaction")
        
    if submitted:
        inputs = {'Amount': amount, 'time_hour': transaction_hour, 'is_night': 1 if is_night else 0, 'amt_deviation': amt_deviation}
        result = ensemble.predict_credit_card(inputs)
        show_fraud_gauge(result.fraud_probability)
        show_alert_badge(result.alert_level)
        show_result_metrics(result)
        
        st.session_state.history.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'domain': result.domain,
            'fraud_prob': result.fraud_probability,
            'alert_level': result.alert_level,
            'fraud_label': result.fraud_label
        })

with tabs[1]:
    st.header("Insurance Claim Analyzer")
    with st.form("ins_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            deductible = st.number_input("Deductible", min_value=0, value=500)
            months_as_customer = st.number_input("Months as Customer", min_value=0, value=24)
            number_of_past_complaints = st.number_input("Number of Past Complaints", min_value=0, value=0)
        with col2:
            incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"])
            collision_type = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision", "?"])
            incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"])
            authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "None"])
        submitted = st.form_submit_button("Analyze Claim")
        
    if submitted:
        # Pass dummy numerical encoding for prediction as real one is complex for GUI 
        inputs = {'age': age, 'deductible': deductible, 'months_as_customer': months_as_customer, 'number_of_past_complaints': number_of_past_complaints, 'incident_type': incident_type, 'collision_type': collision_type, 'incident_severity': incident_severity, 'authorities_contacted': authorities_contacted, 'high_deductible': 1 if deductible>=700 else 0, 'multiple_claims': 1 if number_of_past_complaints>1 else 0, 'recent_claim': 1 if months_as_customer<6 else 0, 'young_driver': 1 if age<=25 else 0}
        
        # NOTE: Predict on raw string is handled internally or mock handled
        result = ensemble.predict_insurance(inputs)
        show_fraud_gauge(result.fraud_probability)
        show_alert_badge(result.alert_level)
        st.session_state.history.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'domain': result.domain,
            'fraud_prob': result.fraud_probability,
            'alert_level': result.alert_level,
            'fraud_label': result.fraud_label
        })

with tabs[2]:
    st.header("E-Commerce Order Checker")
    with st.form("eco_form"):
        col1, col2 = st.columns(2)
        with col1:
            transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=55.0)
            account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
            transaction_hour = st.slider("Transaction Hour", 0, 23, 10, key='eco_hr')
        with col2:
            is_new_account = st.checkbox("Is New Account?")
            address_mismatch = st.checkbox("Address Mismatch?")
            is_high_value = st.checkbox("Is High Value Order?")
            is_unusual_hour = st.checkbox("Is Unusual Hour?")
        submitted = st.form_submit_button("Check Order")
        
    if submitted:
        inputs = {
            'TransactionAmount': transaction_amount,
            'AccountAgeDays': account_age_days,
            'TransactionHour': transaction_hour,
            'is_new_account': 1 if is_new_account else 0,
            'address_mismatch': 1 if address_mismatch else 0,
            'is_high_value': 1 if is_high_value else 0,
            'is_unusual_hour': 1 if is_unusual_hour else 0
        }
        result = ensemble.predict_ecommerce(inputs)
        show_fraud_gauge(result.fraud_probability)
        show_alert_badge(result.alert_level)
        st.session_state.history.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'domain': result.domain,
            'fraud_prob': result.fraud_probability,
            'alert_level': result.alert_level,
            'fraud_label': result.fraud_label
        })

with tabs[3]:
    st.header("Analytics")
    
    total_checked = len(st.session_state.history)
    fraud_detected = sum(1 for item in st.session_state.history if item['fraud_label'])
    avg_risk = sum(item['fraud_prob'] for item in st.session_state.history) / total_checked if total_checked > 0 else 0
    high_risk = sum(1 for item in st.session_state.history if item['alert_level'] == 'High')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Checked", total_checked)
    col2.metric("Fraud Detected", fraud_detected)
    col3.metric("Avg Risk Score", f"{avg_risk:.2%}")
    col4.metric("High Risk Count", high_risk)
    
    st.subheader("Prediction History")
    if total_checked > 0:
        history_df = pd.DataFrame(st.session_state.history).tail(20)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions made yet.")
        
    st.subheader("Global Model Evaluation (ROC)")
    roc_plot_path = os.path.join(PLOTS_DIR, 'roc_all_models.png')
    f1_plot_path = os.path.join(PLOTS_DIR, 'f1_comparison.png')
    
    if os.path.exists(f1_plot_path):
        st.image(f1_plot_path, caption='F1 Score Comparison')
    else:
        st.info("F1 chart not found.")
