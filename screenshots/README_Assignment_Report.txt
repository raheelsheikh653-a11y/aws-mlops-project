========================================================================================
                  AWS MLOps: Zero to Mastery — Practical Assignment Report
                          Student Name: Raheel (Raheel-AWS-Learning)
========================================================================================

Tamam 15 Screenshots sequence ke sath is folder me save kar diye gaye hain:

----------------------------------------------------------------------------------------
MODULE 1: AWS FOUNDATIONS FOR ML
----------------------------------------------------------------------------------------
01_Module1_IAM_Role_Setup.jpeg
   - Description: IAM Role (SageMaker-MLOps-ExecutionRole) creation with trust policy 
     (sagemaker.amazonaws.com) and AmazonSageMakerFullAccess policy.

02_Module1_S3_Bucket_Dataset_Upload.jpeg
   - Description: S3 Bucket (s3://mlops-raheel-bucket-2026/data/) with successfully 
     uploaded customer churn dataset files (train.csv, test.xlsx).

03_Module1_CloudShell_CLI_Verification.jpeg
   - Description: AWS CloudShell terminal executing 'aws s3 ls' verifying uploaded 
     datasets inside the cloud storage bucket.

----------------------------------------------------------------------------------------
MODULE 2: EC2 — YOUR CLOUD COMPUTER
----------------------------------------------------------------------------------------
04_Module2_EC2_IAM_Instance_Role.jpeg
   - Description: IAM Role (Raheel-EC2-S3-Role) configured with AmazonS3FullAccess 
     permission attached to EC2 instance profile.

05_Module2_EC2_Model_Training_S3_Upload.jpeg
   - Description: Linux terminal inside EC2 instance downloading dataset from S3, 
     training RandomForest ML model, and uploading model.joblib back to S3 bucket.

----------------------------------------------------------------------------------------
MODULE 3: SAGEMAKER — ML PLATFORM OF AWS
----------------------------------------------------------------------------------------
06_Module3_SageMaker_Studio_Dashboard.jpeg
   - Description: Active Amazon SageMaker Unified Studio main dashboard overview 
     under project admin-project-197545187149.

07_Module3_Service_Quotas_Management.jpeg
   - Description: AWS Service Quotas console showing quota increase request submitted 
     for ml.m5.large training job instance usage.

08_Module3_SageMaker_Studio_Model_Training.jpeg
   - Description: SageMaker Studio Jupyter Notebook executing model training with 
     97.88% Accuracy and packaging model.tar.gz artifact to S3.

09_Module3_SageMaker_Endpoint_InService.jpeg
   - Description: SageMaker Inference Endpoints table displaying 'raheel-churn-endpoint' 
     with status 'In service' (green checkmark).

10_Module3_Live_REST_API_Prediction_Response.jpeg
   - Description: Testing the deployed REST API endpoint via boto3 runtime sending test 
     customer features and receiving live prediction response: {"prediction": [0]}.

----------------------------------------------------------------------------------------
MODULE 4: AUTOMATION WITH PIPELINES
----------------------------------------------------------------------------------------
11_Module4_SageMaker_Pipeline_DAG_Canvas.jpeg
   - Description: SageMaker Studio Workflows visual DAG canvas showing automated 
     MLOps Pipeline ('Raheel-MLOps-Pipeline') with connected Notebook and Python task operators.

12_Module4_SageMaker_Model_Registry.jpeg
   - Description: SageMaker Model Registry console showing 'Raheel-CustomerChurn-Group' 
     registered with Model Package Version 1 (Approved).

----------------------------------------------------------------------------------------
MODULE 5: MONITORING & MODEL HEALTH
----------------------------------------------------------------------------------------
13_Module5_CloudWatch_Monitoring_Dashboard.jpeg
   - Description: AWS CloudWatch Dashboard ('Raheel-MLOps-Dashboard') displaying real-time 
     MemoryUtilization graph for the deployed endpoint.

14_Module5_AWS_Budgets_Cost_Alarm.jpeg
   - Description: AWS Budgets console showing active $1.00 cost safety alarm 
     ('Raheel-MLOps-Budget') with status OK.

----------------------------------------------------------------------------------------
MODULE 6: CI/CD FOR ML (MLOPS COMPLETE)
----------------------------------------------------------------------------------------
15_Module6_PyTest_ML_Unit_Tests_Passed.jpeg
   - Description: Windows PowerShell terminal at project directory executing 
     'python -m pytest' running automated data quality and model accuracy threshold tests 
     (All 4 tests PASSED).

========================================================================================
Submission Status: 100% COMPLETE & VERIFIED
========================================================================================
