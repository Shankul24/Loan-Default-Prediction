import streamlit as st
import pandas as pd
import joblib
import os

# Construct the path to pickled files using the script's directory
script_dir = os.path.dirname(__file__)
encoder_path = os.path.join(script_dir, "encoder.pkl")
scaler_path = os.path.join(script_dir, "scaler.pkl")
model_path = os.path.join(script_dir, "Loan_default_model.pkl")


# Load the pickled encoder, scaler, and model
encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)
xgb_model = joblib.load(model_path)

# Streamlit app
st.title("Loan Default Prediction App")

# User input fields
person_age = st.number_input("Person Age:", min_value=18, max_value=80, value=30)
person_income = st.number_input("Person Income:", min_value=0, value=50000)
person_emp_length = st.number_input("Person Employment Length (in years):", min_value=0, value=5)
loan_amnt = st.number_input("Loan Amount:", min_value=0, value=10000)
loan_int_rate = st.number_input("Loan Interest Rate (%):", min_value=0.0, value=10.0, step=0.1)
loan_percent_income = st.number_input("Loan Percent Income (%):", min_value=0.0, value=0.2, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length:", min_value=0, value=5)
person_home_ownership = st.selectbox("Person Home Ownership:", ["RENT", "MORTGAGE", "OWN"])
loan_intent = st.selectbox("Loan Intent:", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade:", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.selectbox("Person Default on File:", ["Y", "N"])

# Button to trigger prediction
predict_button = st.button("Show Results")

if predict_button:
    # Preprocess user inputs
    user_inputs = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income/100],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "cb_person_default_on_file": [cb_person_default_on_file]
    })

    # Preprocess user inputs using the encoder and scaler
    encoded_features = encoder.transform(user_inputs[["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]]).toarray()
    scaled_numerical = scaler.transform(user_inputs[["person_age", "person_income", "person_emp_length", "loan_amnt",
                                                 "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]])

    # Convert encoded features to a DataFrame with descriptive column names
    encoded_feature_names = encoder.get_feature_names_out(["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"])
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Convert scaled numerical features to a DataFrame with descriptive column names
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=["person_age", "person_income", "person_emp_length", "loan_amnt",
                                                               "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"])

    # Concatenate the encoded and scaled features DataFrames
    preprocessed_data = pd.concat([encoded_features_df, scaled_numerical_df], axis=1)


    # Predict loan status
    prediction = xgb_model.predict(preprocessed_data)

    # Display prediction result
    st.markdown("### Predicted Loan Status:")
    if prediction == 0:
        st.markdown("<p style='font-size:24px;color:green;'>No Default</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:24px;color:red;'>Default</p>", unsafe_allow_html=True)
