import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and scaler
model = joblib.load('rf_multioutput_model.pkl')
scaler = joblib.load('scaler.pkl')
# Optionally, load the correct column order if dummies were used
try:
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
except:
    feature_columns = None  # If not used

# UI
st.title("Eastern Hararghe Weather Forecast")

st.write("Fill in the features below to predict temperature, rainfall, humidity, windspeed, and precipitation.")

# List your required feature columns (example, adjust as per your model!)
input_features = [
    'GWETPROF', 'GWETTOP', 'GWETROOT', 'CLOUD_AMT', 'TS', 'PS', 'QV2M', 
    # ... more continuous features ...
    # Categorical variables (district, zone, etc.) can be drop-downs:
    'district', 'WeredaCode', 'Zone', 'Region'
]

# For illustration, text inputs for continuous vars, selectbox for categories
user_input = {}
for col in input_features:
    if col in ['district', 'Zone', 'Region']:
        user_input[col] = st.text_input(f"Enter {col}")
    else:
        user_input[col] = st.number_input(f"Enter {col}", value=0.0)

# On Predict button
if st.button('Predict'):
    # Build DataFrame for input
    X_pred = pd.DataFrame([user_input])
    # One-hot-encode as during training
    X_pred = pd.get_dummies(X_pred)
    # Make sure column order matches training
    if feature_columns:
        for col in feature_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_columns]
    # Scale
    X_pred_scaled = scaler.transform(X_pred)
    # Predict
    output = model.predict(X_pred_scaled)[0]
    st.write(f"Predicted Temperature: {output[0]:.2f}")
    st.write(f"Predicted Rainfall: {output[1]:.2f}")
    st.write(f"Predicted Humidity: {output[2]:.2f}")
    st.write(f"Predicted Windspeed: {output[3]:.2f}")
    st.write(f"Predicted Precipitation: {output[4]:.2f}")
