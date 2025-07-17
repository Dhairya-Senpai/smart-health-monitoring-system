

import streamlit as st
import os
import pandas as pd
import json
from encryption import encrypt_data, decrypt_data
from ml_random_forest import train_random_forest
from preprocessing import load_data
from sklearn.preprocessing import StandardScaler


st.title("Secure Report Submission & Feedback (Role-Based Demo)")


# Key for encryption (in production, store securely!)
KEY_PATH = "data/report_key.bin"
REPORTS_DIR = "data/reports"
FEEDBACK_DIR = "data/feedback"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Generate/load encryption key
if os.path.exists(KEY_PATH):
    with open(KEY_PATH, "rb") as f:
        key = f.read()
else:
    key = os.urandom(32)
    with open(KEY_PATH, "wb") as f:
        f.write(key)

# Sample users and assignments
patients = ["Alice", "Bob", "Charlie"]
doctors = ["Dr. Smith", "Dr. Lee"]
nurses = ["Nurse Joy", "Nurse Sam"]
nurse_assignments = {"Nurse Joy": "Dr. Smith", "Nurse Sam": "Dr. Lee"}

# Role selection
st.sidebar.header("User Login")
role = st.sidebar.selectbox("Select your role", ["Patient", "Doctor", "Nurse"])
if role == "Patient":
    user_name = st.sidebar.selectbox("Select your name", patients)
    assigned_doctor = st.sidebar.selectbox("Select your doctor", doctors)
elif role == "Doctor":
    user_name = st.sidebar.selectbox("Select your name", doctors)
elif role == "Nurse":
    user_name = st.sidebar.selectbox("Select your name", nurses)
    assigned_doctor = nurse_assignments[user_name]



st.header("1. Submit a Report or Vitals")
option = st.radio("What are you submitting?", ("Structured Vitals (for ML)", "Unstructured Report (text or file)"))

if role == "Patient":
    if option == "Structured Vitals (for ML)":
        st.subheader("Enter Patient Vitals")
        features = ['age', 'trestbps', 'thalach', 'chol', 'oldpeak']
        defaults = {'age': 55, 'trestbps': 130, 'thalach': 150, 'chol': 240, 'oldpeak': 1.0}
        user_input = {f: st.number_input(f.capitalize(), value=float(defaults[f])) for f in features}
        if st.button("Submit Vitals for Prediction"):
            # ML prediction
            input_df = pd.DataFrame([user_input])
            df = load_data("data/uci_heart_disease.csv")
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=features + ['target'])
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])
            input_scaled = scaler.transform(input_df)
            X = df[features]
            y = (df['target'] > 0).astype(int)
            model = train_random_forest(X, y)
            pred = model.predict(input_scaled)[0]
            feedback = f"ML Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}"
            st.success(feedback)
            # Encrypt and store patient input with metadata
            metadata = {
                "role": "Patient",
                "patient": user_name,
                "doctor": assigned_doctor,
                "type": "ml_vitals"
            }
            input_record = {
                "metadata": metadata,
                "data": user_input
            }
            enc_input = encrypt_data(json.dumps(input_record), key)
            input_name = os.path.join(REPORTS_DIR, f"ml_input_{len(os.listdir(REPORTS_DIR))+1}_{user_name}_{assigned_doctor}.enc")
            with open(input_name, "wb") as f:
                f.write(enc_input)
            st.info(f"Encrypted patient input saved as {os.path.basename(input_name)}")
            # Encrypt and store feedback with metadata
            fb_record = {
                "metadata": metadata,
                "feedback": feedback
            }
            enc_feedback = encrypt_data(json.dumps(fb_record), key)
            fb_name = os.path.join(FEEDBACK_DIR, f"ml_feedback_{len(os.listdir(FEEDBACK_DIR))+1}_{user_name}_{assigned_doctor}.enc")
            with open(fb_name, "wb") as f:
                f.write(enc_feedback)
            st.info(f"Encrypted feedback saved as {os.path.basename(fb_name)}")

    elif option == "Unstructured Report (text or file)":
        st.subheader("Submit a Report (text or file)")
        report_text = st.text_area("Paste your report (or write here):")
        report_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        if st.button("Submit Report for AI Feedback"):
            if report_file:
                content = report_file.read().decode(errors='ignore')
            elif report_text.strip():
                content = report_text
            else:
                st.warning("Please enter text or upload a file.")
                st.stop()
            # Simple rule-based feedback (placeholder for NLP/LLM)
            if any(word in content.lower() for word in ["pain", "chest", "hypertension", "angina"]):
                feedback = "AI Feedback: Symptoms suggest possible cardiac risk. Recommend further evaluation."
            else:
                feedback = "AI Feedback: No immediate cardiac risk detected in report."
            st.success(feedback)
            # Encrypt and store patient report with metadata
            metadata = {
                "role": "Patient",
                "patient": user_name,
                "doctor": assigned_doctor,
                "type": "ai_report"
            }
            input_record = {
                "metadata": metadata,
                "data": content
            }
            enc_input = encrypt_data(json.dumps(input_record), key)
            input_name = os.path.join(REPORTS_DIR, f"ai_input_{len(os.listdir(REPORTS_DIR))+1}_{user_name}_{assigned_doctor}.enc")
            with open(input_name, "wb") as f:
                f.write(enc_input)
            st.info(f"Encrypted patient report saved as {os.path.basename(input_name)}")
            # Encrypt and store feedback with metadata
            fb_record = {
                "metadata": metadata,
                "feedback": feedback
            }
            enc_feedback = encrypt_data(json.dumps(fb_record), key)
            fb_name = os.path.join(FEEDBACK_DIR, f"ai_feedback_{len(os.listdir(FEEDBACK_DIR))+1}_{user_name}_{assigned_doctor}.enc")
            with open(fb_name, "wb") as f:
                f.write(enc_feedback)
            st.info(f"Encrypted feedback saved as {os.path.basename(fb_name)}")

else:
    st.info("Report submission is only available for Patients.")




# --- Real-Time Prediction Section ---
st.header("2. Real-Time Heart Disease Risk Prediction (Trained Models)")
import joblib
rf_model_path = "models/random_forest_model.joblib"
nn_model_path = "models/neural_net_model.joblib"
model_loaded = False
if os.path.exists(rf_model_path) and os.path.exists(nn_model_path):
    rf_model = joblib.load(rf_model_path)
    nn_model = joblib.load(nn_model_path)
    model_loaded = True
else:
    st.warning("Trained models not found. Please run the training pipeline first.")

features = ['age', 'trestbps', 'thalach', 'chol', 'oldpeak']
st.subheader("Enter Patient Vitals for Prediction")
input_data = {}
for f in features:
    input_data[f] = st.number_input(f.capitalize(), value=55.0 if f=='age' else 130.0)
input_df = pd.DataFrame([input_data])

if st.button("Predict with Trained Models") and model_loaded:
    # For demo, use scaler from training data
    df = load_data("data/uci_heart_disease.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=features + ['target'])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    input_scaled = scaler.transform(input_df)
    rf_pred = rf_model.predict(input_scaled)[0]
    nn_pred = nn_model.predict(input_scaled)[0]
    st.success(f"Random Forest Prediction: {'Heart Disease' if rf_pred == 1 else 'No Heart Disease'}")
    st.success(f"Neural Net Prediction: {'Heart Disease' if nn_pred == 1 else 'No Heart Disease'}")
    st.info(f"Encrypted RF prediction: {encrypt_data(str(rf_pred), key).hex()}")
    st.info(f"Encrypted NN prediction: {encrypt_data(str(nn_pred), key).hex()}")

st.header("3. Review Encrypted Data (Role-Based Access)")
# Review patient input
input_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".enc")]
selected_input = st.selectbox("Select patient input to review", input_files, key="input")
if selected_input:
    with open(os.path.join(REPORTS_DIR, selected_input), "rb") as f:
        enc_input = f.read()
    try:
        decrypted_input = decrypt_data(enc_input, key)
        input_record = json.loads(decrypted_input)
        meta = input_record["metadata"]
        # Access control logic
        allowed = False
        if role == "Doctor" and user_name == meta["doctor"]:
            allowed = True
        elif role == "Patient" and user_name == meta["patient"]:
            allowed = True
        elif role == "Nurse" and assigned_doctor == meta["doctor"]:
            allowed = True
        if allowed:
            st.text_area("Decrypted Patient Input/Report", json.dumps(input_record, indent=2), height=150)
        else:
            st.error("Access denied: You are not authorized to view this data.")
    except Exception as e:
        st.error(f"Could not decrypt patient input: {e}")

# Review feedback
feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".enc")]
selected_fb = st.selectbox("Select feedback to review", feedback_files, key="fb")
if selected_fb:
    with open(os.path.join(FEEDBACK_DIR, selected_fb), "rb") as f:
        enc_fb = f.read()
    try:
        decrypted_fb = decrypt_data(enc_fb, key)
        fb_record = json.loads(decrypted_fb)
        meta = fb_record["metadata"]
        allowed = False
        if role == "Doctor" and user_name == meta["doctor"]:
            allowed = True
        elif role == "Patient" and user_name == meta["patient"]:
            allowed = True
        elif role == "Nurse" and assigned_doctor == meta["doctor"]:
            allowed = True
        if allowed:
            st.text_area("Decrypted Feedback", json.dumps(fb_record, indent=2), height=100)
        else:
            st.error("Access denied: You are not authorized to view this feedback.")
    except Exception as e:
        st.error(f"Could not decrypt feedback: {e}")
