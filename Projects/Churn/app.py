import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained pipeline
try:
    loaded_pipeline = joblib.load('churn_prediction_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: 'churn_prediction_pipeline.joblib' not found. Please ensure the pipeline saving cell was executed.")
    st.stop()

st.title('Telco Customer Churn Prediction')
st.write('Enter customer details to predict churn.')

# Input features based on the X_full_pipeline structure
# Numerical features
tenure = st.slider('Tenure (Months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0, step=0.1)
total_charges = st.number_input('Total Charges', min_value=0.0, value=1000.0, step=0.1)

# Categorical features - using actual values before encoding
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Create a DataFrame from inputs
input_data = pd.DataFrame([
    {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
])

# Ensure the order of columns matches the training data used for the pipeline
# This is crucial for ColumnTransformer to work correctly.
# We can get the order from X_full_pipeline's columns, which was used to train the pipeline.
# Assuming X_full_pipeline's columns are available (e.g., from a kernel restart or explicit definition).
# In a real deployment, you'd save these column names alongside the model.

# For demonstration, let's manually define column order based on X_full_pipeline's structure before the transformer
expected_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Reindex input_data to match the expected column order
input_data = input_data[expected_columns]

if st.button('Predict Churn'):
    try:
        prediction = loaded_pipeline.predict(input_data)
        prediction_proba = loaded_pipeline.predict_proba(input_data)

        churn_status = 'Yes' if prediction[0] == 1 else 'No'
        prob_churn = prediction_proba[0][1] * 100
        prob_no_churn = prediction_proba[0][0] * 100

        st.subheader('Prediction Result:')
        if churn_status == 'Yes':
            st.error(f'This customer is likely to CHURN! (Probability: {prob_churn:.2f}%)')
        else:
            st.success(f'This customer is likely to NOT CHURN. (Probability: {prob_no_churn:.2f}%)')

        st.write(f'Probability of Churn: {prob_churn:.2f}%')
        st.write(f'Probability of No Churn: {prob_no_churn:.2f}%')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")