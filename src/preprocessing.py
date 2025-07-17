# Data preprocessing functions for Smart Health Monitoring System

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Add more features from UCI Heart Disease dataset if available
    features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    # Only keep features that exist in the dataframe
    features = [f for f in features if f in df.columns]
    # Clean numeric columns: convert to numeric, coerce errors, drop NaNs
    for f in features:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    df = df.dropna(subset=features)
    # Remove outliers (simple z-score method)
    for f in features:
        if df[f].dtype in [float, int]:
            z = (df[f] - df[f].mean()) / df[f].std()
            df = df[(z.abs() < 3)]
    # Feature engineering: add BMI if height/weight available
    if 'height' in df.columns and 'weight' in df.columns:
        df['BMI'] = df['weight'] / ((df['height']/100)**2)
        features.append('BMI')
    # Age group feature
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0,1,2,3])
        features.append('age_group')
    # Normalize features
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
