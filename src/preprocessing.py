# Data preprocessing functions for Smart Health Monitoring System

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    features = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'spo2']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
# ...existing code...

if __name__ == "__main__":
    df = load_data("data/sample_vitals.csv")
    print("Original Data:")
    print(df)
    df_processed = preprocess_data(df)
    print("\nPreprocessed Data:")
    print(df_processed)
