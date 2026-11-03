"""
AWS Lambda Function: Trigger SageMaker Pipeline on S3 Upload or EventBridge Event
"""
import json
import boto3

sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    pipeline_name = "MLOps-Churn-Pipeline"
    
    try:
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName="Automated-Trigger-Execution",
            PipelineParameters=[
                {
                    'Name': 'InputDataUrl',
                    'Value': 's3://mlops-project-aws-user2026/data/churn_full.csv'
                }
            ]
        )
        print("Pipeline started successfully:", response['PipelineExecutionArn'])
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'SageMaker Pipeline triggered successfully!',
                'execution_arn': response['PipelineExecutionArn']
            })
        }
    except Exception as e:
        print("Error starting pipeline:", str(e))
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
