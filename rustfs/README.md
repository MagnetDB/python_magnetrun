# RustFS Integration for python-magnetrun

Local S3-compatible object storage using [RustFS](https://rustfs.com/) (a MinIO-compatible
server written in Rust), with the `magnetfs` Python package to convert, upload, read, and
plot magnetrun sensor data.

## Overview

The workflow is:
1. Start RustFS locally via Docker Compose
2. Convert raw `.txt` magnetrun data files to Parquet format and upload to RustFS
3. Read and visualize the data directly from RustFS

## Directory Structure

```
rustfs/
├── pyproject.toml           # magnetfs package metadata and entry point
├── magnetfs/                # Python package
│   ├── __init__.py
│   ├── client.py            # Shared S3 client and config
│   ├── conversion.py        # .txt → Parquet conversion and upload
│   ├── storage.py           # List and read Parquet files from bucket
│   ├── plotting.py          # Download and plot Parquet data
│   └── cli.py               # Unified CLI entry point
├── app_streamlit.py         # Streamlit web dashboard
├── app_panel.py             # Panel web dashboard
├── docker-compose.yml       # RustFS server + tester containers
├── Dockerfile               # Python 3.11 image
├── requirements.txt         # Python dependencies (for Docker / manual install)
├── test_scripts.md          # Usage notes for legacy test_*.py scripts
├── test_conversion.py       # Legacy: convert and upload (kept for reference)
├── test_read.py             # Legacy: list and read (kept for reference)
├── test_plot.py             # Legacy: plot (kept for reference)
├── test_upload.py           # Legacy: simple upload pipeline (kept for reference)
├── rustfs_internal_data/    # RustFS data volume (Docker mount)
├── rustfs_logs/             # RustFS log volume (Docker mount)
└── rustfs_sandbox/          # Scratch space
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11+

## Installation

Install the `magnetfs` package from the `rustfs/` directory:

```bash
pip install -e .
```

This installs all required dependencies (`polars`, `boto3`, `botocore`, `matplotlib`) and
registers the `magnetfs` CLI command.

For the web dashboards, install optional dependencies:

```bash
pip install -e ".[dashboard]"
```

## Quick Start

### 1. Start the RustFS server

```bash
docker compose up -d rustfs
```

The RustFS S3 API is available at `http://localhost:9000` and the web console at
`http://localhost:9001`.

Default credentials (development only):
- Access key: `test_user`
- Secret key: `test_password`

### 2. Convert and upload a data file

```bash
# List available .txt files in your data directory
magnetfs --datadir /path/to/data files --ext txt

# Convert a .txt file to Parquet and upload to RustFS
magnetfs --datadir /path/to/data convert M10_2020.10.23---20_10_41.txt
```

The conversion handles the magnetrun `.txt` format (whitespace-separated, first row
skipped) and combines `Date` and `Time` columns into a single `timestamp` column.

### 3. Read data back from RustFS

```bash
# List files in the bucket
magnetfs list

# Download and preview a Parquet file
magnetfs read M10_2020.10.23---20_10_41.parquet
```

### 4. Plot data

```bash
# List columns in a file
magnetfs plot M10_2020.10.23---20_10_41.parquet --keys

# Plot a specific column against time
magnetfs plot M10_2020.10.23---20_10_41.parquet --x timestamp --y Tin1

# Save plot to a PNG file instead of displaying it
magnetfs plot M10_2020.10.23---20_10_41.parquet --x timestamp --y Tin1 --output plot.png

# Interactive mode (prompts for file and column selection)
magnetfs plot
```

If no display server is available (remote server, CI), use `--output plot.png`.

## Web-based Dashboards

Two interactive web dashboards are available as an alternative to the CLI plot command.
Both run entirely in a browser — no display server needed.

### Streamlit ([app_streamlit.py](app_streamlit.py))

```bash
# Locally
streamlit run app_streamlit.py

# Via Docker Compose
docker compose up -d streamlit
```

Open [http://localhost:8501](http://localhost:8501).

Use the sidebar to select a Parquet file, X/Y columns, and optionally show raw data and
statistics.

### Panel ([app_panel.py](app_panel.py))

```bash
# Locally
panel serve app_panel.py --show

# Via Docker Compose
docker compose up -d panel
```

Open [http://localhost:5006/app_panel](http://localhost:5006/app_panel).

> Both dashboards use `matplotlib` with the `Agg` (non-interactive) backend.

## Environment Variables

| Variable          | Default                 | Description            |
|-------------------|-------------------------|------------------------|
| `RUSTFS_ENDPOINT` | `http://localhost:9000` | RustFS S3 endpoint URL |
| `ACCESS_KEY`      | `test_user`             | S3 access key          |
| `SECRET_KEY`      | `test_password`         | S3 secret key          |

## Legacy Scripts

The original `test_*.py` standalone scripts are kept for reference. See
[test_scripts.md](test_scripts.md) for their usage.
