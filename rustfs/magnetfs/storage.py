from io import BytesIO

import polars as pl

from .client import BUCKET_NAME, get_s3_client, list_bucket_files


def read_parquet(target_file: str) -> None:
    """List bucket contents then download and preview a Parquet file."""
    s3 = get_s3_client()

    files = list_bucket_files(s3)
    if not files:
        return

    print(f"\nReading {target_file}...")
    try:
        obj_response = s3.get_object(Bucket=BUCKET_NAME, Key=target_file)
        df = pl.read_parquet(BytesIO(obj_response["Body"].read()))

        print("Data successfully retrieved!")
        print("\n--- Data Preview (First 5 Rows) ---")
        print(df.head())
        print(f"\nDataFrame Shape: {df.shape}")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error reading {target_file}: {e}")
