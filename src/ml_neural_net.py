# Neural Network ML implementation for Smart Health Monitoring System
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


from ml_utils import tune_mlp, cross_validate_model, balance_data_smote

def train_mlp(X, y):
    # Balance data
    X_res, y_res = balance_data_smote(X, y)
    # Hyperparameter tuning and regularization
    best_model, best_params, best_score = tune_mlp(X_res, y_res)
    # Cross-validation
    cv_score, cv_scores = cross_validate_model(best_model, X_res, y_res)
    print(f"Best MLP params: {best_params}, GridSearchCV best score: {best_score}, CV mean: {cv_score}")
    best_model.fit(X_res, y_res)
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm
# ...existing code...

if __name__ == "__main__":
    import sys
    sys.path.append("./src")
    from preprocessing import load_data, preprocess_data

    # Load and preprocess data
    df = load_data("data/sample_vitals.csv")
    df = preprocess_data(df)

    # Simulate a binary target (risk: 0=low, 1=high)
    df['risk'] = ((df['heart_rate'] > 100) | (df['spo2'] < 92)).astype(int)

    X = df[['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'spo2']]
    y = df['risk']

    # Train/test split (use all for train due to small sample)
    model = train_mlp(X, y)
    acc, cm = evaluate_model(model, X, y)
    print("MLP Neural Network Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
