"""
Module 1: AWS Foundations for ML
Creates S3 bucket and uploads dataset
"""
import boto3
import sys

def setup_s3(bucket_name, region="us-east-1"):
    s3_client = boto3.client("s3", region_name=region)
    
    print(f"[*] Creating bucket: {bucket_name} in {region}...")
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print(f"[+] Bucket '{bucket_name}' created successfully!")
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"[!] Bucket '{bucket_name}' already owned by you. Proceeding.")
    except Exception as e:
        print(f"[-] Error: {e}")
        return

    # Upload files
    files = ["train.csv", "test.csv", "churn_full.csv"]
    for file in files:
        try:
            s3_key = f"data/{file}"
            s3_client.upload_file(file, bucket_name, s3_key)
            print(f"[+] Uploaded {file} to s3://{bucket_name}/{s3_key}")
        except FileNotFoundError:
            print(f"[-] Local file '{file}' not found. Run 'python data_generator.py' first.")

    print("\nVerify in CLI using:")
    print(f"  aws s3 ls s3://{bucket_name}/data/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_s3.py <unique-bucket-name> [region]")
        print("Example: python setup_s3.py mlops-aws-bucket-2026 us-east-1")
    else:
        bucket = sys.argv[1]
        reg = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
        setup_s3(bucket, reg)
