import streamlit as st
import joblib
import pickle
import numpy as np
import os

# Title and sidebar with colorful background
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(90deg, #00FFFF 0%, #FF00FF 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and sidebar
st.title("Fraudulent Footprints: Anomaly Detection in Credit Card Transactions")

st.sidebar.title('Input New Transaction Information')

# Input fields
v2 = st.sidebar.number_input(label='V2', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v3 = st.sidebar.number_input(label='V3', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v4 = st.sidebar.number_input(label='V4', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v7 = st.sidebar.number_input(label='V7', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v10 = st.sidebar.number_input(label='V10', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v11 = st.sidebar.number_input(label='V11', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v12 = st.sidebar.number_input(label='V12', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v14 = st.sidebar.number_input(label='V14', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v16 = st.sidebar.number_input(label='V16', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v17 = st.sidebar.number_input(label='V17', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v18 = st.sidebar.number_input(label='V18', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")
v21 = st.sidebar.number_input(label='V21', min_value=-10.0, max_value=10.0, step=0.001, format="%.7f")

# Model selection in a light rectangle
st.subheader('Select Model to Use:')
with st.empty():
    selection = st.radio("", ["Logistic Regression", "Best Model!"])

# Determine the absolute path to the models
current_dir = os.path.dirname(os.path.abspath(__file__))
logreg_model_path = os.path.join(current_dir, "..", "Deployed model", "logreg_model.sav")
svc_model_path = os.path.join(current_dir, "..", "Deployed model", "svc_model.sav")
rf_model_path = os.path.join(current_dir, "..", "Deployed model", "best_model.sav")

# Load selected model
model = None
try:
    if selection == 'Logistic Regression':
        st.markdown('You selected Logistic Regression Model')
        if os.path.exists(logreg_model_path):
            model = joblib.load(logreg_model_path)
        else:
            st.error(f"Model file not found: {logreg_model_path}")
    
    if selection == 'Best Model!':
        st.markdown('You Selected The Best Model!')
        if os.path.exists(svc_model_path):
            model = joblib.load(svc_model_path)
        else:
            st.error(f"Model file not found: {svc_model_path}")
   
except ModuleNotFoundError as e:
    st.error(f"Module not found error: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Button to check transaction
with st.empty():
    if st.button("Check Transaction", key="check_transaction", help="Click to check the transaction"):
        if model:
            result = model.predict(np.array([[v14, v10, v12, v4, v11, v17, v16, v7, v3, v2, v21, v18]]))
            if result[0] == 0:
                st.success("Transaction is safe")
            else: 
                st.error("Transaction is fraudulent!")
        else:
            st.error("Model not loaded. Please check the model file and try again.")
