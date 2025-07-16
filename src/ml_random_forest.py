# Random Forest ML implementation for Smart Health Monitoring System

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

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
    # For demo: risk if heart_rate > 100 or spo2 < 92
    df['risk'] = ((df['heart_rate'] > 100) | (df['spo2'] < 92)).astype(int)

    X = df[['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'spo2']]
    y = df['risk']

    # Train/test split (use all for train due to small sample)
    model = train_random_forest(X, y)
    acc, cm = evaluate_model(model, X, y)
    print("Random Forest Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
