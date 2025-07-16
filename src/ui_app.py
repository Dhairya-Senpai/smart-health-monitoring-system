import streamlit as st
import numpy as np
import pandas as pd
import os
from encryption import encrypt_data, decrypt_data
from ml_random_forest import train_random_forest
from preprocessing import load_data
from sklearn.preprocessing import StandardScaler

st.title("Smart Health Monitoring System - Demo UI")

st.write("Enter patient vital signs to predict heart disease risk. Data is encrypted before prediction.")

# Input fields for features
defaults = {'age': 55, 'trestbps': 130, 'thalach': 150, 'chol': 240, 'oldpeak': 1.0}
features = ['age', 'trestbps', 'thalach', 'chol', 'oldpeak']
user_input = {}
for f in features:
    user_input[f] = st.number_input(f.capitalize(), value=float(defaults[f]))

# Prepare input for model
input_df = pd.DataFrame([user_input])

# Load and preprocess training data, train model (for demo)
df = load_data("data/uci_heart_disease.csv")
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=features + ['target'])
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
input_scaled = scaler.transform(input_df)
X = df[features]
y = (df['target'] > 0).astype(int)  # Binary classification
model = train_random_forest(X, y)

# Encrypt input
key = os.urandom(32)
record_str = ','.join([str(user_input[f]) for f in features])
encrypted = encrypt_data(record_str, key)
st.write("Encrypted input (hex):", encrypted.hex())

# Decrypt for prediction
rec = decrypt_data(encrypted, key)
vals = [float(v) for v in rec.split(',')]
vals_scaled = scaler.transform([vals])

# Predict
if st.button("Predict Risk"):
    pred = model.predict(vals_scaled)[0]
    st.success(f"Predicted risk: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
    encrypted_pred = encrypt_data(str(pred), key)
    st.write("Encrypted prediction (hex):", encrypted_pred.hex())
