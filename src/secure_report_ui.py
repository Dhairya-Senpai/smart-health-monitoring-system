

import streamlit as st
import os
import pandas as pd
import json
from encryption import encrypt_data, decrypt_data
from ml_random_forest import train_random_forest
from preprocessing import load_data
from sklearn.preprocessing import StandardScaler

import datetime
import uuid



import streamlit as st
import os
import pandas as pd
import json
from encryption import encrypt_data, decrypt_data
from ml_random_forest import train_random_forest
from preprocessing import load_data
from sklearn.preprocessing import StandardScaler
import joblib
import PyPDF2

st.set_page_config(page_title="Smart Health Monitoring System", layout="wide")
st.title("Smart Health Monitoring System")

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

st.sidebar.header("User Login")

# --- Session State Reset on Role Change ---
if 'last_role' not in st.session_state:
    st.session_state['last_role'] = None
role = st.sidebar.selectbox("Select your role", ["Patient", "Doctor", "Nurse"])
if st.session_state['last_role'] != role:
    # Reset all relevant input fields
    # Removed direct assignment to widget keys to avoid Streamlit warning
    st.session_state['patient_report_text'] = ''
    st.session_state['patient_report_file'] = None
    st.session_state['patient_radio'] = 'Structured Vitals (form)'
    st.session_state['doctor_input'] = None
    st.session_state['doctor_fb'] = None
    st.session_state['nurse_input'] = None
    st.session_state['nurse_fb'] = None
st.session_state['last_role'] = role

if role == "Patient":
    user_name = st.sidebar.selectbox("Select your name", patients)
    assigned_doctor = st.sidebar.selectbox("Select your doctor", doctors)
elif role == "Doctor":
    user_name = st.sidebar.selectbox("Select your name", doctors)
elif role == "Nurse":
    user_name = st.sidebar.selectbox("Select your name", nurses)
    assigned_doctor = nurse_assignments[user_name]


# --- Conditional Workflow Display ---
if role == "Patient":
    # ...existing code...
    # Assign/generate persistent patient_id for each patient
    if 'patient_ids' not in st.session_state:
        st.session_state['patient_ids'] = {}
    if user_name not in st.session_state['patient_ids']:
        st.session_state['patient_ids'][user_name] = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_name))
    patient_id = st.session_state['patient_ids'][user_name]

    # --- Patient Workflow ---
    st.header("Patient Workflow")
    st.subheader("1. Submit a Report or Vitals")
    option = st.radio("Choose input type:", ("Structured Vitals (form)", "Unstructured Report (text/file upload)"), key="patient_radio")
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak']
    defaults = {'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 130, 'chol': 240, 'thalach': 150, 'oldpeak': 1.0}
    prediction = None
    feedback = None
    input_record = None
    content = None
    report_file = None
    report_text = None
    extracted_text = None
    extracted_fields = {}
    if option == "Structured Vitals (form)":
        user_input = {}
        user_input['age'] = st.number_input('Age', value=float(defaults['age']), key='vitals_age')
        user_input['sex'] = st.selectbox('Sex', options=[0,1], format_func=lambda x: 'Female' if x==0 else 'Male', key='vitals_sex')
        user_input['cp'] = st.selectbox('Chest Pain Type', options=[0,1,2,3], format_func=lambda x: ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'][x], key='vitals_cp')
        user_input['trestbps'] = st.number_input('Resting Blood Pressure', value=float(defaults['trestbps']), key='vitals_trestbps')
        user_input['chol'] = st.number_input('Cholesterol', value=float(defaults['chol']), key='vitals_chol')
        user_input['thalach'] = st.number_input('Max Heart Rate', value=float(defaults['thalach']), key='vitals_thalach')
        user_input['oldpeak'] = st.number_input('ST Depression', value=float(defaults['oldpeak']), key='vitals_oldpeak')
        if st.button("Run Prediction", key="run_structured_pred"):
            input_df = pd.DataFrame([user_input], columns=features)
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
            prediction = 'Heart Disease' if pred == 1 else 'No Heart Disease'
            feedback = f"ML Prediction: {prediction}"
            st.success(feedback)
            input_record = {
                "type": "ml_vitals",
                "data": user_input
            }
    elif option == "Unstructured Report (text/file upload)":
        report_text = st.text_area("Paste your report (or write here):", key="patient_report_text")
        report_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="patient_report_file")
        if report_file and report_file.name.lower().endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(report_file)
                extracted_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
                # Extract only model features from text
                import re
                feature_patterns = {
                    'age': r'age\s*[:=]?\s*(\d+)',
                    'sex': r'sex\s*[:=]?\s*(male|female|0|1)',
                    'cp': r'chest pain type\s*[:=]?\s*(\d+)',
                    'trestbps': r'resting blood pressure\s*[:=]?\s*(\d+)',
                    'chol': r'cholesterol\s*[:=]?\s*(\d+)',
                    'thalach': r'max heart rate\s*[:=]?\s*(\d+)',
                    'oldpeak': r'st depression\s*[:=]?\s*([\d\.]+)'
                }
                for feat, pat in feature_patterns.items():
                    match = re.search(pat, extracted_text, re.IGNORECASE)
                    if match:
                        extracted_fields[feat] = match.group(1)
                if extracted_fields:
                    st.info("Extracted model features from PDF:")
                    for k, v in extracted_fields.items():
                        st.write(f"{k.capitalize()}: {v}")
                else:
                    st.warning("No model features found in PDF text.")
            except Exception as e:
                st.warning(f"Could not extract text from PDF: {e}")
        if st.button("Run AI Feedback", key="run_unstructured_pred"):
            if report_file:
                if report_file.name.lower().endswith('.pdf') and extracted_text:
                    content = extracted_text
                else:
                    content = report_file.read().decode(errors='ignore')
            elif report_text and report_text.strip():
                content = report_text
            else:
                st.warning("Please enter text or upload a file.")
                st.stop()
            st.info("Content used for AI feedback:")
            st.text_area("Content for Feedback", content, height=200)
            if any(word in content.lower() for word in ["pain", "chest", "hypertension", "angina"]):
                feedback = "AI Feedback: Symptoms suggest possible cardiac risk. Recommend further evaluation."
            else:
                feedback = "AI Feedback: No immediate cardiac risk detected in report."
            st.success(feedback)
            input_record = {
                "type": "ai_report",
                "data": content
            }
    elif option == "Unstructured Report (text/file upload)":
        report_text = st.text_area("Paste your report (or write here):", key="patient_report_text")
        report_file = st.file_uploader("Or upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="patient_report_file")
        extracted_text = None
        extracted_fields = {}
        if report_file and report_file.name.lower().endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(report_file)
                extracted_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
                # Extract only model features from text
                import re
                feature_patterns = {
                    'age': r'age\s*[:=]?\s*(\d+)',
                    'sex': r'sex\s*[:=]?\s*(male|female|0|1)',
                    'cp': r'chest pain type\s*[:=]?\s*(\d+)',
                    'trestbps': r'resting blood pressure\s*[:=]?\s*(\d+)',
                    'chol': r'cholesterol\s*[:=]?\s*(\d+)',
                    'thalach': r'max heart rate\s*[:=]?\s*(\d+)',
                    'oldpeak': r'st depression\s*[:=]?\s*([\d\.]+)'
                }
                for feat, pat in feature_patterns.items():
                    match = re.search(pat, extracted_text, re.IGNORECASE)
                    if match:
                        extracted_fields[feat] = match.group(1)
                if extracted_fields:
                    st.info("Extracted model features from PDF:")
                    for k, v in extracted_fields.items():
                        st.write(f"{k.capitalize()}: {v}")
                else:
                    st.warning("No model features found in PDF text.")
            except Exception as e:
                st.warning(f"Could not extract text from PDF: {e}")
        if st.button("Run AI Feedback", key="run_unstructured_pred"):
            if report_file:
                if report_file.name.lower().endswith('.pdf') and extracted_text:
                    content = extracted_text
                else:
                    content = report_file.read().decode(errors='ignore')
            elif report_text.strip():
                content = report_text
            else:
                st.warning("Please enter text or upload a file.")
                st.stop()
            st.info("Content used for AI feedback:")
            st.text_area("Content for Feedback", content, height=200)
            if any(word in content.lower() for word in ["pain", "chest", "hypertension", "angina"]):
                feedback = "AI Feedback: Symptoms suggest possible cardiac risk. Recommend further evaluation."
            else:
                feedback = "AI Feedback: No immediate cardiac risk detected in report."
            st.success(feedback)
            input_record = {
                "type": "ai_report",
                "data": content
            }

    # Always show submit button
    if st.button("Submit to Doctor", key="submit_to_doctor"):
        # Determine what to submit based on current input type
        if option == "Structured Vitals (form)":
            submit_data = {
                "type": "ml_vitals",
                "data": {k: st.session_state.get(f"vitals_{k}", None) for k in features}
            }
        else:
            # For unstructured, prefer extracted PDF fields if available
            if report_file and report_file.name.lower().endswith('.pdf') and extracted_fields:
                submit_data = {
                    "type": "ai_report",
                    "data": extracted_fields
                }
            else:
                submit_data = {
                    "type": "ai_report",
                    "data": report_text if report_text.strip() else None
                }
        metadata = {
            "role": "Patient",
            "patient": user_name,
            "patient_id": patient_id,
            "doctor": assigned_doctor,
            "type": submit_data["type"]
        }
        full_record = {
            "metadata": metadata,
            "data": submit_data["data"]
        }
        enc_input = encrypt_data(json.dumps(full_record), key)
        import datetime
        # Parse first and last name
        name_parts = user_name.split()
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        input_base = f"{first_name}_{last_name}_{timestamp}"
        input_name = os.path.join(REPORTS_DIR, f"{input_base}.enc")
        with open(input_name, "wb") as f:
            f.write(enc_input)
        st.info(f"Encrypted input saved as {os.path.basename(input_name)}")
        # If PDF was uploaded, encrypt and save with matching base name
        if report_file and report_file.name.lower().endswith('.pdf'):
            pdf_save_path = os.path.join(REPORTS_DIR, f"{input_base}.pdf.enc")
            report_file.seek(0)
            pdf_bytes = report_file.read()
            enc_pdf = encrypt_data(pdf_bytes, key)
            with open(pdf_save_path, "wb") as pdf_out:
                pdf_out.write(enc_pdf)
            st.info(f"Encrypted PDF saved as {os.path.basename(pdf_save_path)}")
        # If feedback is available, submit it too
        if feedback:
            fb_record = {
                "metadata": metadata,
                "feedback": feedback
            }
            enc_feedback = encrypt_data(json.dumps(fb_record), key)
            fb_name = os.path.join(FEEDBACK_DIR, f"feedback_{len(os.listdir(FEEDBACK_DIR))+1}_{user_name}_{assigned_doctor}.enc")
            with open(fb_name, "wb") as f:
                f.write(enc_feedback)
            st.info(f"Encrypted feedback saved as {os.path.basename(fb_name)}")

elif role == "Doctor":
    st.header("Doctor Workflow")
    st.subheader("Review Patient Inputs and Feedback")
    # Gather all reports and feedback, group by patient_id
    input_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".enc")]
    patient_map = {}
    report_map = {}
    for fname in input_files:
        try:
            with open(os.path.join(REPORTS_DIR, fname), "rb") as f:
                enc_input = f.read()
            decrypted_input = decrypt_data(enc_input, key)
            input_record = json.loads(decrypted_input)
            meta = input_record.get("metadata", {})
            pid = meta.get("patient_id", None)
            pname = meta.get("patient", None)
            if pid:
                patient_map[pid] = pname
                if pid not in report_map:
                    report_map[pid] = []
                report_map[pid].append((fname, input_record))
        except Exception:
            continue
    # Doctor selects patient by name/ID
    patient_options = [f"{patient_map[pid]} ({pid[:8]})" for pid in patient_map]
    selected_patient = st.selectbox("Select patient to review", patient_options, key="doctor_patient_search")
    # Extract patient_id from selection
    selected_pid = None
    if selected_patient:
        for pid in patient_map:
            if selected_patient.endswith(f"({pid[:8]})"):
                selected_pid = pid
                break
    # Show all reports for selected patient
    if selected_pid and selected_pid in report_map:
        st.markdown(f"### Reports for {patient_map[selected_pid]} ({selected_pid[:8]})")
        for fname, input_record in sorted(report_map[selected_pid], key=lambda x: x[0], reverse=True):
            meta = input_record["metadata"]
            allowed = user_name == meta["doctor"]
            if allowed:
                st.markdown(f"---\n**Report File:** {fname}")
                st.markdown("**Type:** {}  ".format(meta.get("type", "")))
                data = input_record.get("data", {})
                # Try to show PDF if original was PDF
                pdf_path = None
                fname_base = os.path.basename(fname).replace(".enc", "")
                for f in os.listdir(REPORTS_DIR):
                    if f.endswith(".pdf.enc") and fname_base in f:
                        pdf_path = os.path.join(REPORTS_DIR, f)
                        break
                if pdf_path and os.path.exists(pdf_path):
                    import base64
                    st.markdown("**Original PDF Report (Encrypted):**")
                    with open(pdf_path, "rb") as pdf_file:
                        enc_pdf_bytes = pdf_file.read()
                        try:
                            pdf_bytes = decrypt_data(enc_pdf_bytes, key)
                            st.download_button("Download PDF", pdf_bytes, file_name=os.path.basename(pdf_path).replace('.enc',''))
                            pdf_base64 = base64.b64encode(pdf_bytes).decode()
                            st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="400"></iframe>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Could not decrypt PDF: {e}")
                if isinstance(data, dict):
                    st.markdown("**Report Data:**")
                    df = pd.DataFrame([data])
                    st.table(df)
                else:
                    st.markdown("**Report Content:**")
                    st.text_area("Report Text", str(data), height=150)

                # Doctor feedback and medication input
                st.markdown("**Doctor Feedback & Prescription**")
                feedback_text = st.text_area("Enter feedback/notes", key=f"doctor_feedback_{fname}")
                meds_text = st.text_area("Enter medications/prescription", key=f"doctor_meds_{fname}")
                if st.button("Submit Feedback & Prescription", key=f"submit_doctor_feedback_{fname}"):
                    fb_record = {
                        "metadata": meta,
                        "feedback": feedback_text,
                        "medications": meds_text,
                        "report_file": fname
                    }
                    enc_feedback = encrypt_data(json.dumps(fb_record), key)
                    fb_name = os.path.join(FEEDBACK_DIR, f"doctorfb_{len(os.listdir(FEEDBACK_DIR))+1}_{meta['patient']}_{meta['doctor']}.enc")
                    with open(fb_name, "wb") as f:
                        f.write(enc_feedback)
                    st.success(f"Feedback & prescription saved as {os.path.basename(fb_name)}")
            else:
                st.error("Access denied: You are not authorized to view this data.")
    # Show feedback for selected patient
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".enc")]
    fb_map = []
    for fname in feedback_files:
        try:
            with open(os.path.join(FEEDBACK_DIR, fname), "rb") as f:
                enc_fb = f.read()
            decrypted_fb = decrypt_data(enc_fb, key)
            fb_record = json.loads(decrypted_fb)
            meta = fb_record.get("metadata", {})
            pid = meta.get("patient_id", None)
            if pid == selected_pid:
                fb_map.append((fname, fb_record))
        except Exception:
            continue
    if fb_map:
        st.markdown(f"### Feedback for {patient_map[selected_pid]} ({selected_pid[:8]})")
        for fname, fb_record in sorted(fb_map, key=lambda x: x[0], reverse=True):
            st.markdown(f"---\n**Feedback File:** {fname}")
            st.text_area("Decrypted Feedback", json.dumps(fb_record, indent=2), height=100, key=f"doctor_fb_view_{fname}")
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".enc")]
    selected_fb = st.selectbox("Select feedback to review", feedback_files, key="doctor_fb")
    if selected_fb:
        with open(os.path.join(FEEDBACK_DIR, selected_fb), "rb") as f:
            enc_fb = f.read()
        try:
            decrypted_fb = decrypt_data(enc_fb, key)
            fb_record = json.loads(decrypted_fb)
            meta = fb_record["metadata"]
            allowed = user_name == meta["doctor"]
            if allowed:
                st.text_area("Decrypted Feedback", json.dumps(fb_record, indent=2), height=100)
            else:
                st.error("Access denied: You are not authorized to view this feedback.")
        except Exception as e:
            st.error(f"Could not decrypt feedback: {e}")

elif role == "Nurse":
    st.header("Nurse Workflow")
    st.subheader("Review Assigned Patient Inputs and Feedback")
    input_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".enc")]
    selected_input = st.selectbox("Select patient input to review", input_files, key="nurse_input")
    if selected_input:
        with open(os.path.join(REPORTS_DIR, selected_input), "rb") as f:
            enc_input = f.read()
        try:
            decrypted_input = decrypt_data(enc_input, key)
            input_record = json.loads(decrypted_input)
            meta = input_record["metadata"]
            allowed = assigned_doctor == meta["doctor"]
            if allowed:
                st.text_area("Decrypted Patient Input/Report", json.dumps(input_record, indent=2), height=150)
            else:
                st.error("Access denied: You are not authorized to view this data.")
        except Exception as e:
            st.error(f"Could not decrypt patient input: {e}")
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.endswith(".enc")]
    selected_fb = st.selectbox("Select feedback to review", feedback_files, key="nurse_fb")
    if selected_fb:
        with open(os.path.join(FEEDBACK_DIR, selected_fb), "rb") as f:
            enc_fb = f.read()
        try:
            decrypted_fb = decrypt_data(enc_fb, key)
            fb_record = json.loads(decrypted_fb)
            meta = fb_record["metadata"]
            allowed = assigned_doctor == meta["doctor"]
            if allowed:
                st.text_area("Decrypted Feedback", json.dumps(fb_record, indent=2), height=100)
            else:
                st.error("Access denied: You are not authorized to view this feedback.")
        except Exception as e:
            st.error(f"Could not decrypt feedback: {e}")

# --- Real-Time Prediction Section (All Roles) ---
