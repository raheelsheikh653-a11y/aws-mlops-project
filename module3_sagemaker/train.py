"""
SageMaker Entrypoint Training & Inference Script (train.py)
Executed inside SageMaker container
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def model_fn(model_dir):
    """Loads model for SageMaker Endpoint serving"""
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)

    # SageMaker container environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    args = parser.parse_args()

    train_file = os.path.join(args.train, "train.csv")
    print(f"Reading data from: {train_file}")
    df = pd.read_csv(train_file)

    X_train = df.drop(columns=['churn'])
    y_train = df['churn']

    print(f"Training RandomForest with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Train Accuracy: {train_acc:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
