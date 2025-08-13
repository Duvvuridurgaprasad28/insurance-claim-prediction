import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

# Page title
st.title("🏥 Insurance Claim Prediction App")

# Load the categorical dataset for dropdown options
health_categorical = pd.read_csv("data/health_updated.csv")
# Load the numeric dataset for encoding reference
health_numeric = pd.read_csv("data/X_features.csv")

# Load model and scaler
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Get unique values from categorical data for user-friendly dropdowns
cities = sorted(health_categorical['city'].unique().tolist())
jobs = sorted(health_categorical['job_title_grouped'].unique().tolist())

# Create mapping from categorical to numeric (assuming same order)
city_to_numeric = {city: i for i, city in enumerate(cities)}
job_to_numeric = {job: i for i, job in enumerate(jobs)}

# Sidebar input fields
st.sidebar.header("📋 User Input Features")
age = st.sidebar.slider("Age", 18, 64, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
weight = st.sidebar.slider("Weight (kg)", 40, 130, 70)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
hereditary_diseases = st.sidebar.selectbox("Hereditary Diseases", ["Yes", "No"])
no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])

# User sees city names, but we convert to numeric for the model
city_name = st.sidebar.selectbox("City", cities)
city_code = city_to_numeric[city_name]  # Convert name to number

bloodpressure = st.sidebar.slider("Blood Pressure", 60, 180, 120)
diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"])
regular_ex = st.sidebar.selectbox("Regular Exercise", ["Yes", "No"])

# User sees job names, but we convert to numeric for the model
job_name = st.sidebar.selectbox("Job Title", jobs)
job_code = job_to_numeric[job_name]  # Convert name to number

# Encode the input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Male" else 0],
    'weight': [weight],
    'bmi': [bmi],
    'hereditary_diseases': [1 if hereditary_diseases == "Yes" else 0],
    'no_of_dependents': [no_of_dependents],
    'smoker': [1 if smoker == "Yes" else 0],
    'city': [city_code],  # Use the converted number
    'bloodpressure': [bloodpressure],
    'diabetes': [1 if diabetes == "Yes" else 0],
    'regular_ex': [1 if regular_ex == "Yes" else 0],
    'job_title_grouped': [job_code]  # Use the converted number
})

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

st.subheader("💰 Predicted Insurance Claim Amount")
st.success(f"$ {prediction:,.2f}")

# SHAP Explanation
st.subheader("🔍 SHAP Explanation")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # Create SHAP force plot
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        matplotlib=False
    )

    # Convert SHAP to HTML string
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    # Show in Streamlit
    st.components.v1.html(shap_html, height=400)

except Exception as e:
    st.error(f"SHAP explanation failed: {e}")