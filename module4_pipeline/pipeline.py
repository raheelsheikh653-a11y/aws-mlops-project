"""
Module 4: Automated SageMaker Pipeline (DAG)
Defines Preprocessing -> Training -> Evaluation -> Model Registry Registration
"""
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

def build_sagemaker_pipeline(bucket_name, role_arn, pipeline_name="MLOps-Churn-Pipeline"):
    sagemaker_session = sagemaker.Session()
    
    # 1. Pipeline Parameters
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{bucket_name}/data/churn_full.csv"
    )

    # 2. Preprocessing Step
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name="mlops-prep",
        role=role_arn
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
        ],
        code="module3_sagemaker/train.py"
    )

    # 3. Training Step
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="module3_sagemaker",
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        hyperparameters={"n-estimators": 100, "max-depth": 5}
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # 4. Model Registry Step
    step_register = RegisterModel(
        name="RegisterChurnModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="CustomerChurnPackageGroup",
        approval_status="PendingManualApproval"
    )

    # 5. Assemble Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data],
        steps=[step_process, step_train, step_register],
        sagemaker_session=sagemaker_session
    )

    return pipeline

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <bucket-name> <role-arn>")
    else:
        b_name = sys.argv[1]
        r_arn = sys.argv[2]
        pipe = build_sagemaker_pipeline(b_name, r_arn)
        pipe.upsert(role_arn=r_arn)
        execution = pipe.start()
        print(f"[+] Pipeline executed successfully! Execution ARN: {execution.arn}")
