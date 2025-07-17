
# Smart Health Monitoring System

A secure, ML-powered system for real-time patient vital sign monitoring and risk prediction.

## Features
- Real-time data simulation or ingestion
- Random Forest and Neural Network models for risk prediction
- AES-256 encryption for secure data storage and transmission
- Modular codebase for easy extension
- **Supports integration of multiple heart disease datasets (UCI, Kaggle, custom)**
- Automated dataset combining and model retraining


## Directory Structure
```
SmartHealthMonitoringSystem/
├── data/                # Datasets, encrypted files, combined_heart_disease.csv
├── models/              # Saved ML models
├── src/                 # Source code
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```


## Tech Stack
- Python
- scikit-learn, pandas, numpy
- joblib, imbalanced-learn, kagglehub
- cryptography, pycryptodome
- Streamlit/Flask (optional UI)


## Datasets
- **UCI Heart Disease**: Downloaded and integrated using `ucimlrepo`.
- **Kaggle Heart Disease**: Downloaded manually or via `kagglehub`.
- **Custom/Other CSVs**: Place any compatible heart disease CSVs in the `data/` folder.
- Features used: `age`, `sex`, `trestbps`, `chol`, `thalach`, `oldpeak`, `target`
- Target: `target` (can be multiclass or converted to binary: `target > 0`)


## Workflow
1. **Add datasets**: Place CSVs (UCI, Kaggle, custom) in `data/` folder. If zipped, just drop the zip file.
2. **Combine datasets**: `python src/combine_datasets.py` (automatically extracts zips and merges all suitable CSVs)
3. **Preprocess and train models**: `python src/app.py` (uses combined_heart_disease.csv)
4. **ML prediction**: Random Forest and Neural Network on combined patient data
5. **Encrypted predictions**: For secure storage/transmission


## How to Run
1. Create and activate a Python virtual environment
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add datasets (CSV or zip) to `data/` folder
4. Combine datasets:
   ```
   python src/combine_datasets.py
   ```
5. Train and evaluate models:
   ```
   python src/app.py
   ```


## Security
- All patient data is encrypted using AES-256 before storage or transmission.
- Decryption occurs only for ML prediction and authorized access.


## Notes
- The `target` column may be multiclass (0,1,2,3,4). For binary classification, use `target > 0`.
- You can easily switch to other datasets by adding/removing CSVs in the `data/` folder.
- The pipeline automatically merges all compatible datasets for improved accuracy.

---

For technical details, see `docs/architecture.md`.
