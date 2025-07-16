# Smart Health Monitoring System

A secure, ML-powered system for real-time patient vital sign monitoring and risk prediction.

## Features
- Real-time data simulation or ingestion
- Random Forest and Neural Network models for risk prediction
- AES-256 encryption for secure data storage and transmission
- Modular codebase for easy extension
- **Now supports real UCI Heart Disease dataset integration**

## Directory Structure
```
SmartHealthMonitoringSystem/
├── data/                # Datasets, encrypted files
├── models/              # Saved ML models
├── src/                 # Source code
├── docs/                # Documentation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
```

## Tech Stack
- Python
- scikit-learn, pandas, numpy
- cryptography
- ucimlrepo (for UCI dataset)
- Streamlit/Flask (optional UI)

## Dataset
- **UCI Heart Disease**: Downloaded and integrated using `ucimlrepo`.
- Features used: `age`, `trestbps`, `thalach`, `chol`, `oldpeak`
- Target: `target` (can be multiclass or converted to binary: `target > 0`)

## Workflow
1. **Fetch real dataset**: `python src/fetch_data.py`
2. **Preprocess and encrypt data**: `python src/app.py`
3. **ML prediction**: Random Forest on real patient data
4. **Encrypted predictions**: For secure storage/transmission

## How to Run
1. Create and activate a Python virtual environment
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install ucimlrepo
   ```
3. Fetch the real dataset:
   ```
   python src/fetch_data.py
   ```
4. Run the full pipeline:
   ```
   python src/app.py
   ```

## Security
- All patient data is encrypted using AES-256 before storage or transmission.
- Decryption occurs only for ML prediction and authorized access.

## Notes
- The UCI dataset `target` column may be multiclass (0,1,2,3,4). For binary classification, use `target > 0`.
- You can easily switch to other datasets by updating the data loading and feature selection in `app.py`.

---
For technical details, see `docs/architecture.md`.
