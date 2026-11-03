"""
Module 6: CI/CD ML Unit & Integration Tests (pytest)
Tests data integrity, features, and model performance threshold before deployment
"""
import os
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "train.csv"

@pytest.fixture
def load_data():
    if not os.path.exists(DATA_PATH):
        from data_generator import generate_dataset
        generate_dataset()
    return pd.read_csv(DATA_PATH)

def test_dataset_columns(load_data):
    """Test that all required features exist"""
    expected_cols = {'tenure', 'monthly_charges', 'total_charges', 'contract_type', 'tech_support', 'churn'}
    assert expected_cols.issubset(set(load_data.columns)), "Missing required dataset columns!"

def test_no_missing_values(load_data):
    """Test that there are zero null/NaN values"""
    assert load_data.isnull().sum().sum() == 0, "Dataset contains unexpected NaN values!"

def test_target_distribution(load_data):
    """Ensure target variable contains binary values (0 and 1)"""
    unique_vals = set(load_data['churn'].unique())
    assert unique_vals == {0, 1}, f"Unexpected target classes: {unique_vals}"

def test_model_accuracy_threshold(load_data):
    """Ensure trained model meets minimal production accuracy threshold (> 75%)"""
    df = load_data
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    
    print(f"\nModel Accuracy in CI/CD Test: {acc:.4f}")
    assert acc >= 0.75, f"Model accuracy {acc:.2f} is below production bar of 0.75!"
