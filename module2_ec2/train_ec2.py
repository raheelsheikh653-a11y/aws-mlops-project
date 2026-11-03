"""
Module 2: EC2 Training Script
Downloads data from S3, trains a model on EC2, and saves artifact back to S3.
"""
import os
import sys
import boto3
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_on_ec2(bucket_name, region="us-east-1"):
    s3 = boto3.client('s3', region_name=region)
    local_data = "train.csv"
    
    print(f"[*] Downloading train data from s3://{bucket_name}/data/train.csv ...")
    try:
        s3.download_file(bucket_name, "data/train.csv", local_data)
        print("[+] Download complete!")
    except Exception as e:
        print(f"[!] Could not download from S3 ({e}). Looking for local train.csv...")
        if not os.path.exists(local_data):
            print("[-] train.csv not found! Exiting.")
            return

    # Load data
    df = pd.read_csv(local_data)
    X = df.drop(columns=['churn'])
    y = df['churn']

    print(f"[*] Training RandomForestClassifier on EC2 compute instance...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"[+] Training Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y, y_pred))

    # Save model artifact
    model_filename = "model.joblib"
    joblib.dump(clf, model_filename)
    print(f"[+] Model saved locally as {model_filename}")

    # Upload back to S3
    s3_target_key = "models/ec2-model/model.joblib"
    print(f"[*] Uploading model to s3://{bucket_name}/{s3_target_key} ...")
    try:
        s3.upload_file(model_filename, bucket_name, s3_target_key)
        print(f"[+] SUCCESS! Model saved to S3: s3://{bucket_name}/{s3_target_key}")
    except Exception as e:
        print(f"[-] S3 upload failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_ec2.py <bucket-name> [region]")
    else:
        train_on_ec2(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "us-east-1")
