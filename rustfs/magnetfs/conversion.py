import contextlib
import os
import re
import sys
from io import BytesIO, StringIO

import polars as pl

from .client import BUCKET_NAME, get_s3_client


def list_local_files(datadir: str, exts: set[str] | None = None) -> None:
    """Print files in datadir, optionally filtered by extension(s)."""
    try:
        entries = sorted(os.listdir(datadir))
    except FileNotFoundError:
        print(f"Directory not found: {datadir}")
        sys.exit(1)
    for name in entries:
        if os.path.isfile(os.path.join(datadir, name)) and (
            exts is None or os.path.splitext(name)[1] in exts
        ):
            print(name)


def run_conversion(source_txt: str) -> None:
    """Convert a magnetrun .txt file to Parquet and upload to RustFS."""
    s3 = get_s3_client()

    print(f"Parsing raw text data from {source_txt}...")
    try:
        with open(source_txt) as f:
            lines = f.readlines()

        # skiprows=1: skip first row; normalize whitespace for Polars CSV parser
        content = "".join(lines[1:])
        content = re.sub(r"[^\S\n]+", " ", content)
        content = re.sub(r"^ | $", "", content, flags=re.MULTILINE)

        df = pl.read_csv(StringIO(content), separator=" ", infer_schema_length=1000)
        print(f"Raw data loaded: {len(df)} rows, columns: {df.columns}")

        if "Date" in df.columns and "Time" in df.columns:
            df = df.with_columns(
                (pl.col("Date") + " " + pl.col("Time"))
                .str.to_datetime("%Y.%m.%d %H:%M:%S")
                .alias("timestamp")
            )

        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        with contextlib.suppress(Exception):
            s3.create_bucket(Bucket=BUCKET_NAME)

        parquet_name = os.path.splitext(os.path.basename(source_txt))[0] + ".parquet"
        s3.upload_fileobj(buffer, BUCKET_NAME, parquet_name)
        print(f"Uploaded {parquet_name} ({len(df)} rows)")

    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
