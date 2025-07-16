
import streamlit as st
import os
import pandas as pd
from encryption import encrypt_data, decrypt_data
from ml_random_forest import train_random_forest
from preprocessing import load_data
from sklearn.preprocessing import StandardScaler

st.title("Secure Report Submission & Feedback")

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


st.header("1. Submit a Report or Vitals")
option = st.radio("What are you submitting?", ("Structured Vitals (for ML)", "Unstructured Report (text or file)"))


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
        # Encrypt and store patient input
        input_str = ', '.join([f"{k}: {v}" for k, v in user_input.items()])
        enc_input = encrypt_data(input_str, key)
        input_name = os.path.join(REPORTS_DIR, f"ml_input_{len(os.listdir(REPORTS_DIR))+1}.enc")
        with open(input_name, "wb") as f:
            f.write(enc_input)
        st.info(f"Encrypted patient input saved as {os.path.basename(input_name)}")
        # Encrypt and store feedback
        enc_feedback = encrypt_data(feedback, key)
        fb_name = os.path.join(FEEDBACK_DIR, f"ml_feedback_{len(os.listdir(FEEDBACK_DIR))+1}.enc")
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
        # Encrypt and store patient report
        enc_input = encrypt_data(content, key)
        input_name = os.path.join(REPORTS_DIR, f"ai_input_{len(os.listdir(REPORTS_DIR))+1}.enc")
        with open(input_name, "wb") as f:
            f.write(enc_input)
        st.info(f"Encrypted patient report saved as {os.path.basename(input_name)}")
        # Encrypt and store feedback
        enc_feedback = encrypt_data(feedback, key)
        fb_name = os.path.join(FEEDBACK_DIR, f"ai_feedback_{len(os.listdir(FEEDBACK_DIR))+1}.enc")
        with open(fb_name, "wb") as f:
            f.write(enc_feedback)
        st.info(f"Encrypted feedback saved as {os.path.basename(fb_name)}")



st.header("2. Admin: Review Encrypted Data")
# Review patient input
input_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".enc")]
selected_input = st.selectbox("Select patient input to review", input_files, key="input")
if selected_input:
    with open(os.path.join(REPORTS_DIR, selected_input), "rb") as f:
        enc_input = f.read()
    try:
        decrypted_input = decrypt_data(enc_input, key)
        st.text_area("Decrypted Patient Input/Report", decrypted_input, height=100)
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
        st.text_area("Decrypted Feedback", decrypted_fb, height=100)
    except Exception as e:
        st.error(f"Could not decrypt feedback: {e}")
