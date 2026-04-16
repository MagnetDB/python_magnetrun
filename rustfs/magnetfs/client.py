import os

import boto3
from botocore.config import Config

RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "test_user")
SECRET_KEY = os.getenv("SECRET_KEY", "test_password")
BUCKET_NAME = "dashboard-data"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def list_bucket_files(s3) -> list[str]:
    """List all files in the bucket. Returns empty list if bucket is empty."""
    print(f"Listing files in bucket '{BUCKET_NAME}':")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)

    if "Contents" not in response:
        print("  (Bucket is empty)")
        return []

    files = []
    for obj in response["Contents"]:
        print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        files.append(obj["Key"])
    return files
