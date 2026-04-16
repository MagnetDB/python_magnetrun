import argparse
import os
from io import BytesIO

import boto3
import polars as pl
from botocore.config import Config

# --- CONFIGURATION (Matches your Docker Compose env) ---
RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "test_user")
SECRET_KEY = os.getenv("SECRET_KEY", "test_password")
BUCKET_NAME = "dashboard-data"


def list_bucket_files(s3) -> bool:
    """List all files in the bucket. Returns False if bucket is empty."""
    print(f"Listing files in bucket '{BUCKET_NAME}':")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)

    if "Contents" not in response:
        print("⚠️ Bucket is empty. Run test_conversion.py first!")
        return False

    for obj in response["Contents"]:
        print(f" - Found: {obj['Key']} ({obj['Size']} bytes)")
    return True


def run_reader_test(target_file: str):
    # 1. Initialize S3 Client
    # We use path-style addressing because RustFS (like MinIO) expects it
    s3 = boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    print(f"--- Connecting to RustFS at {RUSTFS_ENDPOINT} ---")

    # 2. List all files in the bucket
    try:
        if not list_bucket_files(s3):
            return

        # 3. Read the specific Parquet file
        print(f"\nReading {target_file} into Polars...")

        obj_response = s3.get_object(Bucket=BUCKET_NAME, Key=target_file)

        # We read the streaming body into a memory buffer
        data_body = obj_response["Body"].read()
        df = pl.read_parquet(BytesIO(data_body))

        # 4. Success Output
        print("✅ Data successfully retrieved!")
        print("\n--- Data Preview (First 5 Rows) ---")
        print(df.head())

        print(f"\nDataFrame Shape: {df.shape}")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"❌ Error during read test: {e}")


def make_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def cmd_list(_args):
    s3 = make_s3_client()
    list_bucket_files(s3)


def cmd_read(args):
    run_reader_test(args.target_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interact with RustFS bucket.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List files in the bucket").set_defaults(func=cmd_list)

    read_parser = subparsers.add_parser("read", help="Read a Parquet file from the bucket")
    read_parser.add_argument("target_file", help="Parquet file key in the bucket to read")
    read_parser.set_defaults(func=cmd_read)

    args = parser.parse_args()
    args.func(args)
