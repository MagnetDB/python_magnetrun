import argparse
import contextlib
import os
import re
import sys
from io import BytesIO, StringIO

import boto3
import polars as pl
from botocore.config import Config

RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "test_user")
SECRET_KEY = os.getenv("SECRET_KEY", "test_password")
BUCKET_NAME = "dashboard-data"


def run_conversion(source_txt: str):
    # 1. Setup S3 Client
    s3 = boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    # 2. Process the weird .txt format
    print(f"Parsing raw text data from {source_txt}...")
    try:
        # Equivalent to pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1):
        # Polars doesn't support regex separators, so normalize whitespace first.
        with open(source_txt) as f:
            lines = f.readlines()
        # skiprows=1: skip first row, second row becomes header
        content = "".join(lines[1:])
        # Replace any run of whitespace (except newlines) with a single space
        content = re.sub(r"[^\S\n]+", " ", content)
        # Strip leading/trailing spaces on each line
        content = re.sub(r"^ | $", "", content, flags=re.MULTILINE)

        df = pl.read_csv(
            StringIO(content),
            separator=" ",
            infer_schema_length=1000,
        )
        print(f"✅ Raw data loaded with {len(df)} rows and columns: {df.columns}")
        print(f"df.keys(): {df.columns}")

        # Optimization: Combine Date and Time into a single datetime column
        if "Date" in df.columns and "Time" in df.columns:
            df = df.with_columns(
                (pl.col("Date") + " " + pl.col("Time"))
                .str.to_datetime("%Y.%m.%d %H:%M:%S")
                .alias("timestamp")
            )

        # 3. Convert to Parquet and Upload
        print("Uploading optimized Parquet...")
        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        with contextlib.suppress(Exception):
            s3.create_bucket(Bucket=BUCKET_NAME)

        parquet_name = os.path.splitext(os.path.basename(source_txt))[0] + ".parquet"
        s3.upload_fileobj(buffer, BUCKET_NAME, parquet_name)
        print(f"✅ Success! Processed {len(df)} rows.")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"❌ Parser Error: {e}")
        # If standard read fails, we might need to use a more manual cleanup
        sys.exit(1)


def cmd_list(args):
    exts = {e if e.startswith(".") else f".{e}" for e in args.ext} if args.ext else None
    try:
        entries = sorted(os.listdir(args.datadir))
    except FileNotFoundError:
        print(f"Directory not found: {args.datadir}")
        sys.exit(1)
    for name in entries:
        if os.path.isfile(os.path.join(args.datadir, name)) and (
            exts is None or os.path.splitext(name)[1] in exts
        ):
            print(name)


def cmd_convert(args):
    source_path = os.path.join(args.datadir, args.source_txt)
    run_conversion(source_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage sensor data files and upload to RustFS."
    )
    parser.add_argument(
        "--datadir",
        default="/mnt/sshfs_data",
        help="Directory to search for data files (default: /mnt/sshfs_data)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List files in datadir")
    list_parser.add_argument(
        "--ext",
        nargs="+",
        metavar="EXT",
        help="Filter by file extension(s), e.g. --ext txt csv",
    )
    list_parser.set_defaults(func=cmd_list)

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a .txt file to Parquet and upload to RustFS"
    )
    convert_parser.add_argument(
        "source_txt", help="Filename to convert (relative to datadir)"
    )
    convert_parser.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)
