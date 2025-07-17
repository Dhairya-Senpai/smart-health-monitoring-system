# Main application entry point for Smart Health Monitoring System

import os
from preprocessing import load_data, preprocess_data
from ml_random_forest import train_random_forest, evaluate_model
from encryption import encrypt_data, decrypt_data
import pandas as pd



def main():
    # Step 1: Load and preprocess combined Heart Disease data
    df = load_data("data/combined_heart_disease.csv")
    df.columns = [c.strip() for c in df.columns]
    target = 'target'
    # Use improved preprocessing
    df = preprocess_data(df)
    # Select features (all except target)
    features = [c for c in df.columns if c != target]
    df = df.dropna(subset=features + [target])

    # Step 2: Encrypt each patient record (features only)
    key = os.urandom(32)  # AES-256 key
    encrypted_records = []
    for _, row in df.iterrows():
        record_str = ','.join([str(row[f]) for f in features])
        encrypted = encrypt_data(record_str, key)
        encrypted_records.append(encrypted)

    print("Encrypted patient records (hex):")
    for enc in encrypted_records[:5]:
        print(enc.hex())

    # Step 3: Decrypt records for ML prediction
    decrypted_data = []
    for enc in encrypted_records:
        rec = decrypt_data(enc, key)
        vals = rec.split(',')
        decrypted_data.append({f: float(v) for f, v in zip(features, vals)})
    df_decrypted = pd.DataFrame(decrypted_data)

    # Step 4: ML prediction and evaluation
    X = df_decrypted[features]
    y = df[target].astype(int)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X, y)
    rf_acc, rf_cm = evaluate_model(rf_model, X, y)
    print(f"Random Forest Accuracy: {rf_acc}")
    print(f"Random Forest Confusion Matrix:\n{rf_cm}")

    print("\nTraining Neural Network...")
    from ml_neural_net import train_mlp, evaluate_model as eval_mlp
    nn_model = train_mlp(X, y)
    nn_acc, nn_cm = eval_mlp(nn_model, X, y)
    print(f"Neural Net Accuracy: {nn_acc}")
    print(f"Neural Net Confusion Matrix:\n{nn_cm}")

    # Step 5: Optionally encrypt predictions
    rf_predictions = rf_model.predict(X)
    nn_predictions = nn_model.predict(X)
    encrypted_rf_preds = [encrypt_data(str(pred), key).hex() for pred in rf_predictions[:10]]
    encrypted_nn_preds = [encrypt_data(str(pred), key).hex() for pred in nn_predictions[:10]]
    print("\nEncrypted RF predictions (hex):", encrypted_rf_preds)
    print("Encrypted NN predictions (hex):", encrypted_nn_preds)

    # Step 6: Save models
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest_model.joblib")
    joblib.dump(nn_model, "models/neural_net_model.joblib")
    print("\nModels saved to models/ directory.")

if __name__ == "__main__":
    main()
