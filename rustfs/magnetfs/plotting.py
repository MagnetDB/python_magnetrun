import os
from io import BytesIO

import matplotlib.pyplot as plt
import polars as pl

from .client import BUCKET_NAME, get_s3_client, list_bucket_files


def _load_parquet(s3, filename: str) -> pl.DataFrame:
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    return pl.read_parquet(BytesIO(obj["Body"].read()))


def plot_data(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    filename: str,
    output: str | None = None,
) -> None:
    """Plot y_col vs x_col from df. Saves to file if output is given."""
    plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], marker=".", linestyle="-", alpha=0.6, color="#2c3e50")
    plt.title(f"{os.path.splitext(filename)[0]}\n{y_col} vs {x_col}", fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


def run_plot(
    filename: str | None,
    x_col: str | None,
    y_col: str | None,
    keys: bool,
    output: str | None,
) -> None:
    """Download a Parquet file from RustFS and plot two columns."""
    s3 = get_s3_client()

    if filename is None:
        files = list_bucket_files(s3)
        if not files:
            return
        choice = input("\nEnter file number or name: ").strip()
        try:
            filename = files[int(choice)] if choice.isdigit() else choice
        except (ValueError, IndexError):
            filename = choice

    print(f"Downloading {filename}...")
    df = _load_parquet(s3, filename)

    if keys:
        print(f"\nColumns in '{filename}':")
        for col in df.columns:
            print(f"  {col} ({df[col].dtype})")
        return

    print("\nColumns found:", df.columns)
    field_x = x_col or input("Select X-axis (default 'Time'): ").strip() or "Time"
    field_y = y_col or input("Select Y-axis (e.g., 'Tin1'): ").strip()

    if field_x in df.columns and field_y in df.columns:
        plot_data(df, field_x, field_y, filename, output=output)
    else:
        print("Error: columns not found. Check spelling (case-sensitive).")
