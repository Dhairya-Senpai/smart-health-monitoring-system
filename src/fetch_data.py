# Script to fetch real datasets for Smart Health Monitoring System


import os
import pandas as pd

# Ensure data directory exists
data_dir = os.path.join('..', 'data')
os.makedirs(data_dir, exist_ok=True)

# Fetch UCI Heart Disease dataset using ucimlrepo
try:
    from ucimlrepo import fetch_ucirepo
    print("Fetching UCI Heart Disease dataset...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    X['target'] = y
    out_path = os.path.join('..', 'data', 'uci_heart_disease.csv')
    X.to_csv(out_path, index=False)
    print(f"Saved UCI Heart Disease dataset to {out_path}")
except Exception as e:
    print("Could not fetch UCI Heart Disease dataset:", e)

# Template for Kaggle Heart Failure dataset using mlcroissant
try:
    import mlcroissant as mlc
    print("Fetching Kaggle Heart Failure dataset (template)...")
    croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/croissant/download')
    record_sets = croissant_dataset.metadata.record_sets
    print("Record sets:", record_sets)
    record_set_df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
    out_path2 = os.path.join('..', 'data', 'kaggle_heart_failure.csv')
    record_set_df.to_csv(out_path2, index=False)
    print(f"Saved Kaggle Heart Failure dataset to {out_path2}")
except Exception as e:
    print("Could not fetch Kaggle Heart Failure dataset:", e)
