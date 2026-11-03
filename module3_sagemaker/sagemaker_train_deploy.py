"""
Module 3: SageMaker Training, Endpoint Deployment & Hyperparameter Tuning (HPO)
Run this script locally or inside SageMaker Studio Notebook
"""
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import IntegerParameter, HyperparameterTuner
import boto3
import sys

def run_sagemaker_workflow(bucket_name, role_arn=None):
    session = sagemaker.Session()
    
    if role_arn is None:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            print("[!] Please provide SageMaker Execution Role ARN as 2nd argument.")
            return
    else:
        role = role_arn

    s3_train_data = f"s3://{bucket_name}/data/train.csv"
    print(f"[*] Training data source: {s3_train_data}")
    print(f"[*] SageMaker Execution Role: {role}")

    # 1. SageMaker Estimator (Training Job)
    sklearn_estimator = SKLearn(
        entry_point="train.py",
        source_dir="module3_sagemaker",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        base_job_name="mlops-churn-rf",
        hyperparameters={"n-estimators": 100, "max-depth": 5}
    )

    print("\n--- [Step 1: Starting SageMaker Training Job] ---")
    sklearn_estimator.fit({"train": s3_train_data})
    print("[+] Training job completed successfully!")

    # 2. Deploy Model as a Real-Time REST API Endpoint
    print("\n--- [Step 2: Deploying Real-time REST Endpoint] ---")
    endpoint_name = "mlops-churn-endpoint"
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=endpoint_name
    )
    print(f"[+] Endpoint deployed successfully: {endpoint_name}")

    # 3. Test Endpoint with sample prediction
    print("\n--- [Step 3: Testing Endpoint with Sample Inference] ---")
    # Sample features: tenure, monthly_charges, total_charges, contract_type, tech_support
    sample_input = [[24, 70.5, 1692.0, 1, 1]]
    prediction = predictor.predict(sample_input)
    print(f"[+] Sample Input: {sample_input}")
    print(f"[+] Endpoint Prediction Response: {prediction}")

    # 4. Hyperparameter Tuning Job (HPO)
    print("\n--- [Step 4: Launching Hyperparameter Tuning (HPO) Job] ---")
    hyperparameter_ranges = {
        "n-estimators": IntegerParameter(50, 200),
        "max-depth": IntegerParameter(3, 10),
    }

    tuner = HyperparameterTuner(
        estimator=sklearn_estimator,
        objective_metric_name="train:accuracy",
        objective_type="Maximize",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=4,
        max_parallel_jobs=2,
        base_tuning_job_name="mlops-churn-hpo"
    )
    print("[*] Starting HPO job...")
    tuner.fit({"train": s3_train_data})
    print("[+] HPO Job initiated successfully! Check AWS SageMaker Console.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sagemaker_train_deploy.py <bucket-name> [role-arn]")
    else:
        b_name = sys.argv[1]
        r_arn = sys.argv[2] if len(sys.argv) > 2 else None
        run_sagemaker_workflow(b_name, r_arn)
