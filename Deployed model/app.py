import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost
import os

# Streamlit app title
st.title("Transaction Fraud Prediction App")

# Display dependency versions for debugging
st.info(f"NumPy version: {np.__version__}")
st.info(f"XGBoost version: {xgboost.__version__}")

# Get the current script's directory and construct model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "xgb_model.pkl")

# Load the fraud detection model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Please ensure 'App/fraud_detection_model.pkl' exists in the project directory.")
    st.stop()
except ImportError as e:
    st.error(f"Dependency error: {e}. Try reinstalling dependencies with: pip install numpy==1.26.4 xgboost")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}. This may be due to a version mismatch. Try: pip install xgboost or re-saving the model in the current environment.")
    st.stop()

# Create a form for user input
with st.form("fraud_prediction_form"):
    st.header("Enter Transaction Details")
    
    # Arrange input fields in a grid layout (3 columns, 2 rows)
    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        tx_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, label_visibility="visible")
    with col2:
        terminal_degree = st.number_input("Terminal Degree", min_value=0, value=5, label_visibility="visible")
    with col3:
        txs_last_1h = st.number_input("Transactions in Last 1 Hour", min_value=0, value=1, label_visibility="visible")

    # Row 2
    col1, col2, col3 = st.columns(3)
    with col1:
        z_score = st.number_input("Transaction Z-Score", value=0.0, label_visibility="visible")
    with col2:
        time_since_last_tx = st.number_input("Time Since Last Transaction (Seconds)", min_value=0, value=3600, label_visibility="visible")
    with col3:
        new_terminal_flag = st.selectbox("New Terminal?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", label_visibility="visible")
    
    # Submit button
    submitted = st.form_submit_button("Predict Fraud")

# Prediction logic
if submitted:
    # Prepare input data
    input_data = pd.DataFrame({
        'TX_AMOUNT': [tx_amount],
        'TERMINAL_DEGREE': [terminal_degree],
        'TXs_LAST_1H': [txs_last_1h],
        'Z_SCORE': [z_score],
        'TIME_SINCE_LAST_TX': [time_since_last_tx],
        'NEW_TERMINAL_FLAG': [new_terminal_flag]
    })
    
    # Make prediction directly (no preprocessing)
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.subheader("Fraud Prediction Result")
        if prediction == 1:
            st.error(f"Fraud Detected! (Confidence: {probability:.2%})")
        else:
            st.success(f"No Fraud Detected (Confidence: {1 - probability:.2%})")
    except Exception as e:
        st.error(f"Error making prediction: {e}. Ensure the model expects 6 raw features (TX_AMOUNT, TERMINAL_DEGREE, TXs_LAST_1H, Z_SCORE, TIME_SINCE_LAST_TX, NEW_TERMINAL_FLAG) and check dependency versions (xgboost).")