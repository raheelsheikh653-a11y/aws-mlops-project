"""
Generates synthetic Customer Churn dataset for AWS MLOps Project
"""
import pandas as pd
import numpy as np

def generate_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(18.0, 118.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 10, size=n_samples), 2)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20]) # 0: Month-to-month, 1: One year, 2: Two year
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    
    # Probability of churn
    logit = -0.05 * tenure + 0.02 * monthly_charges - 1.2 * contract_type - 0.8 * tech_support
    prob = 1 / (1 + np.exp(-logit))
    churn = (prob > 0.45).astype(int)

    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'tech_support': tech_support,
        'churn': churn
    })
    
    # Save train & test splits
    train_df = df.sample(frac=0.8, random_state=random_state)
    test_df = df.drop(train_df.index)

    df.to_csv("churn_full.csv", index=False)
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print(f"Generated {n_samples} samples. Saved churn_full.csv, train.csv, and test.csv")

if __name__ == "__main__":
    generate_dataset()
