"""
Module 5: CloudWatch Dashboards & Monitoring Setup
Automates CloudWatch Dashboard creation for SageMaker Endpoints
"""
import boto3
import json
import sys

def setup_cloudwatch_monitoring(endpoint_name="mlops-churn-endpoint", region="us-east-1"):
    cw = boto3.client('cloudwatch', region_name=region)
    dashboard_name = f"{endpoint_name}-Dashboard"

    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name, "VariantName", "AllTraffic", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": region,
                    "title": f"{endpoint_name} - Invocations Count"
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_name, "VariantName", "AllTraffic", {"stat": "Average"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": region,
                    "title": f"{endpoint_name} - Model Latency (Microseconds)"
                }
            }
        ]
    }

    try:
        cw.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        print(f"[+] CloudWatch Dashboard '{dashboard_name}' created successfully!")
        print(f"[+] View your dashboard in AWS Console -> CloudWatch -> Dashboards -> {dashboard_name}")
    except Exception as e:
        print(f"[-] Error creating dashboard: {e}")

if __name__ == "__main__":
    ep = sys.argv[1] if len(sys.argv) > 1 else "mlops-churn-endpoint"
    reg = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
    setup_cloudwatch_monitoring(ep, reg)
