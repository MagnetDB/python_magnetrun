import argparse
import os
from io import BytesIO

import boto3
import matplotlib.pyplot as plt
import polars as pl
from botocore.config import Config

# --- CONFIG ---
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
        config=Config(signature_version="s3v4"),
    )


def list_available_files(s3):
    """Lists all files in the RustFS bucket."""
    print(f"\n📂 Available files in '{BUCKET_NAME}':")
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" not in response:
            print("   (Bucket is empty)")
            return []

        files = [obj["Key"] for obj in response["Contents"]]
        for i, name in enumerate(files):
            print(f"  [{i}] {name}")
        return files
    except (OSError, RuntimeError) as e:
        print(f"❌ Error listing files: {e}")
        return []


def plot_data(df, x_col, y_col, filename, output=None):
    """Generates the field vs field plot."""
    plt.figure(figsize=(12, 6))

    # Check for timestamp if user wants to plot against Time
    # Note: Ensure your conversion script created the 'timestamp' column!
    plt.plot(
        df[x_col], df[y_col], marker=".", linestyle="-", alpha=0.6, color="#2c3e50"
    )

    plt.title(f"{os.path.splitext(filename)[0]}\n{y_col} vs {x_col}", fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Auto-rotate dates if X is a timestamp
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot data from RustFS parquet files.")
    parser.add_argument(
        "filename", nargs="?", help="File name (or key) in the bucket to load"
    )
    parser.add_argument(
        "--x", default=None, help="X-axis column name (default: 'Time')"
    )
    parser.add_argument("--y", default=None, help="Y-axis column name")
    parser.add_argument(
        "--keys",
        action="store_true",
        help="List available columns in the parquet file and exit",
    )
    parser.add_argument(
        "--output", default=None, help="Save plot to PNG file instead of displaying it"
    )
    args = parser.parse_args()

    s3 = get_s3_client()

    # 1. Search & Select File
    if args.filename:
        filename = args.filename
    else:
        files = list_available_files(s3)
        if not files:
            return
        choice = input("\nEnter the file number or name: ").strip()
        try:
            filename = files[int(choice)] if choice.isdigit() else choice
        except (ValueError, IndexError):
            filename = choice

    # 2. Download into RAM
    print(f"⏳ Downloading {filename}...")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    df = pl.read_parquet(BytesIO(obj["Body"].read()))

    # 3. Optionally just list keys and exit
    if args.keys:
        print(f"\n📊 Columns in '{filename}':")
        for col in df.columns:
            print(f"  {col} ({df[col].dtype})")
        return

    # 4. Choose Fields
    print("\n📊 Columns found:", df.columns)
    field_x = args.x or input("Select X-axis (default 'Time'): ").strip() or "Time"
    field_y = args.y or input("Select Y-axis (e.g., 'Tin1'): ").strip()

    if field_x in df.columns and field_y in df.columns:
        plot_data(df, field_x, field_y, filename, output=args.output)
    else:
        print("❌ Error: Columns not found. Check spelling (case sensitive).")


if __name__ == "__main__":
    main()
