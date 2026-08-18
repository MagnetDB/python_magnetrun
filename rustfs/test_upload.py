import contextlib
from io import BytesIO

import boto3
import polars as pl

# --- CONFIGURATION ---
SSHFS_FILE_PATH = "/path/to/your/sshfs/mount/raw_data.csv"
RUSTFS_ENDPOINT = "http://localhost:9000" # Use NAS IP if running remotely
ACCESS_KEY = "test_user"
SECRET_KEY = "test_password"
BUCKET_NAME = "dashboard-data"

def run_test_pipeline():
    # 1. Connect to RustFS
    s3 = boto3.client(
        's3',
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    # Create bucket if it doesn't exist
    with contextlib.suppress(Exception):
        s3.create_bucket(Bucket=BUCKET_NAME)

    # 2. Convert CSV to Parquet
    print(f"Reading {SSHFS_FILE_PATH}...")
    try:
        # We use Polars for speed and memory efficiency
        df = pl.read_csv(SSHFS_FILE_PATH)
        
        # Save to a memory buffer as Parquet
        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        
        # 3. Upload to RustFS
        print("Uploading Parquet to RustFS...")
        s3.upload_fileobj(buffer, BUCKET_NAME, "optimized_data.parquet")
        print("✅ Success!")
        
    except (OSError, RuntimeError) as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_test_pipeline()

