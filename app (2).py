import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load the saved model, scaler, and feature columns ---
# Using st.cache_resource to cache these heavy objects, preventing them from reloading on every rerun
@st.cache_resource
def load_artifacts():
    try:
        with open('loan_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('feature_columns.pkl', 'rb') as file:
            feature_columns = pickle.load(file)
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure 'loan_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the same directory as this app.py file.")
        st.stop() # Stop the app if files are missing

model, scaler, feature_columns = load_artifacts()

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🏦 Loan Approval Prediction Application")
st.markdown("--- ")
st.write("Enter the applicant's details below to predict their loan approval status. This application uses a machine learning model trained on historical loan data.")

# --- User Input Fields ---
st.header("Applicant Information")

# Layout inputs in two columns for better readability
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=['Male', 'Female'], help="Select the applicant's gender.")
    married = st.selectbox("Married", options=['Yes', 'No'], help="Is the applicant married?")
    dependents = st.selectbox("Dependents", options=['0', '1', '2', '3+'], help="Number of dependents the applicant has.")
    education = st.selectbox("Education", options=['Graduate', 'Not Graduate'], help="Applicant's education level.")
    self_employed = st.selectbox("Self Employed", options=['No', 'Yes'], help="Is the applicant self-employed?")

with col2:
    applicant_income = st.number_input("Applicant Income (USD)", min_value=0, value=5000, step=100, help="Applicant's monthly income.")
    coapplicant_income = st.number_input("Coapplicant Income (USD)", min_value=0, value=0, step=100, help="Coapplicant's monthly income (if any).")
    loan_amount = st.number_input("Loan Amount (USD in thousands)", min_value=1, value=120, step=10, help="Loan amount requested in thousands.") * 1000 # Convert to actual value
    loan_amount_term = st.selectbox("Loan Amount Term (in months)", options=[12, 36, 60, 120, 180, 240, 300, 360, 480], index=7, help="Loan term in months.")
    credit_history = st.selectbox("Credit History", options=[1.0, 0.0], format_func=lambda x: "Yes (1.0)" if x==1.0 else "No (0.0)", help="Does the applicant have a credit history (1.0 for outstanding debts, 0.0 otherwise)?")
    property_area = st.selectbox("Property Area", options=['Urban', 'Rural', 'Semiurban'], help="Location of the property.")

# --- Prediction Button ---
st.markdown("--- ")
if st.button("Predict Loan Status", help="Click to get the loan approval prediction.", type="primary"):
    # --- 2. Create DataFrame from user input (mirroring df_cleaned before encoding) ---
    # Ensure the order and names of columns match the original preprocessing steps
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # --- 3. Preprocessing: One-Hot Encoding ---
    # Identify categorical features to encode (these are the original categorical columns)
    categorical_features_for_encoding = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    # Apply one-hot encoding with drop_first=True, consistent with training
    input_encoded = pd.get_dummies(input_data, columns=categorical_features_for_encoding, drop_first=True)

    # --- 4. Preprocessing: Align columns with training data ---
    # Create a DataFrame with all feature columns from training, initialized to 0
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Populate final_input with values from input_encoded
    # This handles cases where a dummy variable might not be present in a single user input (e.g., only 'Male' chosen, so 'Gender_Female' not created)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col]

    # --- 5. Preprocessing: Apply StandardScaler to numerical features ---
    # Identify numerical features that need scaling
    numerical_features_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    # Apply the pre-fitted scaler to the numerical features of the aligned input
    # Ensure we only try to scale columns that actually exist in the final_input (which they should if feature_columns is correct)
    final_input[numerical_features_to_scale] = scaler.transform(final_input[numerical_features_to_scale])

    # --- 6. Make Prediction ---
    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)

    # --- 7. Display Result ---
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"#### ✅ Loan Approved! (Probability: {prediction_proba[0][1]*100:.2f}%) ")
        st.balloons()
    else:
        st.error(f"#### ❌ Loan Not Approved. (Probability: {prediction_proba[0][0]*100:.2f}%) ")
    
    st.markdown("--- ")
    st.write("**Input Data Used for Prediction (Preprocessed):**")
    st.dataframe(final_input)

st.markdown("--- ")
st.info("Developed by a Machine Learning Enthusiast using Python, Streamlit, and Scikit-learn.")
