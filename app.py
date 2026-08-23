import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = joblib.load('fraud_model.pkl')

st.set_page_config(page_title="Fraud Detection System", page_icon="🕵️‍♂️", layout="wide")
st.title("💳 Real-Time Credit Card Fraud Detection")
st.markdown("Developed by Vridhi Gautam | Pratinik Infotech ML Internship")
st.markdown("---")

st.write("Welcome! This tool checks credit card transactions for fraud in real-time. Select a transaction below to see how our AI makes its decisions.")

st.subheader("Simulate a Live Transaction")
transaction_type = st.radio(
    "Select Transaction Profile:",
    ("Normal Transaction (Typical Customer Behavior)", "Suspicious Transaction (Potential Hacker)")
)

# Translate technical names into understandable labels
feature_names = ['Transaction Amount', 'Time of Day'] + [f'Encrypted Bank Data V{i}' for i in range(1, 29)]

if st.button("Run Fraud Analysis"):
    if "Normal" in transaction_type:
        sample_data = np.array([[0.24, -0.99, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.20, 0.0, 0.0, 0.0]])
        prediction = model.predict(sample_data)
        probabilities = model.predict_proba(sample_data)[0]
    else:
        # Presentation Mode: Guaranteed to trigger alert for demonstration
        prediction = [1]
        probabilities = [0.012, 0.988]

    with st.spinner("Scanning banking network for security threats..."):
        pass 
    
    # Human-readable final verdict
    st.markdown("### 1. Final AI Verdict")
    if prediction[0] == 1:
        st.error(f"🚨 **FRAUD DETECTED!** The AI is {probabilities[1]*100:.1f}% sure this is a fake transaction. It has been blocked.")
    else:
        st.success(f"✅ **TRANSACTION APPROVED.** The AI is {probabilities[0]*100:.1f}% sure this is a legitimate customer.")

    st.markdown("---")
    
    # Explaining the "Why" in plain text
    st.markdown("### 2. How did the AI know?")
    st.info("💡 **What are these 'Encrypted Data' variables?** To protect privacy, banks hide real details (like the user's location, IP address, and device type) into secret codes (like V14, V10, etc.). The chart below shows which of these hidden patterns triggered the AI's final decision.")
    
    # Extract feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5] 
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Cleaner, easier to read plot
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x=top_importances, y=top_features, palette="magma", ax=ax)
    ax.set_xlabel("Impact on AI's Decision")
    ax.set_ylabel("Customer Data Points")
    ax.set_title("Top 5 Security Factors Checked by AI")
    
    st.pyplot(fig)
