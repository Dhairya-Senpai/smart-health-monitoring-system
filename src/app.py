# Main application entry point for Smart Health Monitoring System

import os
from preprocessing import load_data, preprocess_data
from ml_random_forest import train_random_forest, evaluate_model
from encryption import encrypt_data, decrypt_data
import pandas as pd


def main():
    # Step 1: Load and preprocess UCI Heart Disease data

    df = load_data("data/uci_heart_disease.csv")
    # Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]

    # Select features and target
    features = ['age', 'trestbps', 'thalach', 'chol', 'oldpeak']
    target = 'target'
    df = df.dropna(subset=features + [target])

    # Preprocess features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Step 2: Encrypt each patient record (features only)
    key = os.urandom(32)  # AES-256 key
    encrypted_records = []
    for _, row in df.iterrows():
        record_str = ','.join([str(row[f]) for f in features])
        encrypted = encrypt_data(record_str, key)
        encrypted_records.append(encrypted)

    print("Encrypted patient records (hex):")
    for enc in encrypted_records[:5]:  # Show only first 5 for brevity
        print(enc.hex())

    # Step 3: Decrypt records for ML prediction
    decrypted_data = []
    for enc in encrypted_records:
        rec = decrypt_data(enc, key)
        vals = rec.split(',')
        decrypted_data.append({f: float(v) for f, v in zip(features, vals)})
    df_decrypted = pd.DataFrame(decrypted_data)

    # Step 4: ML prediction
    X = df_decrypted[features]
    y = df[target].astype(int)
    model = train_random_forest(X, y)
    predictions = model.predict(X)
    print("\nPredicted target (0=no disease, 1=disease):", predictions[:10])

    # Step 5: Optionally encrypt predictions
    encrypted_preds = [encrypt_data(str(pred), key).hex() for pred in predictions[:10]]
    print("\nEncrypted predictions (hex):", encrypted_preds)

if __name__ == "__main__":
    main()
