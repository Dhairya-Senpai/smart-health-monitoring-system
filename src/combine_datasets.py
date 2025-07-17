# Script to download, preprocess, and combine multiple heart disease datasets



import pandas as pd
import os
import zipfile

# Extract all zip files in data folder
for fname in os.listdir('data'):
    if fname.endswith('.zip'):
        with zipfile.ZipFile(os.path.join('data', fname), 'r') as zip_ref:
            zip_ref.extractall('data')


# Automatically find and combine all CSVs in data folder
common_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
dfs = []
for fname in os.listdir('data'):
    if fname.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join('data', fname))
            df.columns = [c.strip().lower() for c in df.columns]
            # Try to select only the common features
            selected = [f for f in common_features if f in df.columns]
            if len(selected) >= 5:  # Only use if enough features match
                dfs.append(df[selected])
        except Exception as e:
            print(f"Skipping {fname}: {e}")
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna()
    combined.to_csv("data/combined_heart_disease.csv", index=False)
    print(f"Combined dataset saved to data/combined_heart_disease.csv with {len(combined)} rows.")
else:
    print("No suitable CSV files found in data folder.")
