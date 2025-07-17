# Model tuning, cross-validation, and class balancing utilities for Smart Health Monitoring System

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# Hyperparameter tuning for Random Forest

def tune_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# Hyperparameter tuning for MLP

def tune_mlp(X, y):
    param_grid = {
        'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000],
        'early_stopping': [True]
    }
    mlp = MLPClassifier(random_state=42)
    grid = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

# Cross-validation utility

def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return np.mean(scores), scores

# SMOTE for class balancing

def balance_data_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
