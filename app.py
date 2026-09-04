import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from agent import ask_stakeholder_agent

st.set_page_config(page_title="Credit Risk Model Showdown", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return model, scaler, feature_names, metrics

model, scaler, feature_names, metrics = load_assets()

st.title("Credit Risk Model Showdown: XGBoost vs PyTorch MLP")

tab1, tab2 = st.tabs(["📊 Live Model Benchmarks & Scoring", "🤖 Ask the AI Advisor"])

with tab1:
    st.subheader("Benchmark Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Winning Production Model", metrics["winning_model"])
        st.write("**XGBoost AUC-ROC:**", metrics["metrics"]["XGBoost"]["auc"])
        st.write("**XGBoost Latency:**", f"{metrics['metrics']['XGBoost']['latency_ms']} ms")
        
    with col2:
        st.write("**PyTorch MLP AUC-ROC:**", metrics["metrics"]["PyTorch_MLP"]["auc"])
        st.write("**PyTorch MLP Latency:**", f"{metrics['metrics']['PyTorch_MLP']['latency_ms']} ms")

    st.divider()
    st.subheader("Evaluate Applicant Credit Risk")
    
    # Inputs matching dataset numerical fields
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        duration = st.slider("Duration (Months)", 4, 72, 24)
    with col_b:
        amount = st.number_input("Credit Amount", 250, 20000, 5000)
    with col_c:
        age = st.slider("Age (Years)", 18, 75, 30)

    if st.button("Calculate Risk Score"):
        # Create empty row matching your model's expected column list
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        # Assign UI inputs to matching German column names
        if "laufzeit" in input_data.columns:
            input_data["laufzeit"] = duration

        if "hoehe" in input_data.columns:
            input_data["hoehe"] = credit_amount

        if "alter" in input_data.columns:
            input_data["alter"] = age

        # Scale and predict probability
        scaled_data = scaler.transform(input_data)
        prob = model.predict_proba(scaled_data)[0][1]

        # Output dynamic risk score
        if prob >= 0.5:
            st.error(f"High Risk Flagged — Default Probability: {prob * 100:.2f}%")
        else:
            st.success(f"Low Risk Approved — Default Probability: {prob * 100:.2f}%")

with tab2:
    st.subheader("Stakeholder AI Advisor")
    st.caption("Ask questions regarding model selection, AUC metrics, or risk trade-offs.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_prompt := st.chat_input("E.g., Why did we select XGBoost over the PyTorch MLP?"):
        with st.chat_message("user"):
            st.write(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing benchmark data..."):
                response = ask_stakeholder_agent(user_prompt, st.session_state.chat_history)
                st.write(response)
                
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
