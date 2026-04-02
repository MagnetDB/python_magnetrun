"""
Panel demo — plot magnetrun Parquet data from RustFS.

Run locally:
    panel serve app_panel.py --show

Run via Docker:
    docker compose exec tester panel serve app_panel.py --port 5006 --address 0.0.0.0 --allow-websocket-origin=*
"""

import os
from io import BytesIO

import boto3
import matplotlib
import matplotlib.pyplot as plt
import panel as pn
import polars as pl
from botocore.config import Config

matplotlib.use("Agg")  # non-interactive backend — Panel handles the display
pn.extension(sizing_mode="stretch_width")

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


def list_files() -> list[str]:
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]
    except (OSError, RuntimeError) as e:
        print(f"Cannot list files: {e}")
        return []


def load_parquet(filename: str) -> pl.DataFrame:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    return pl.read_parquet(BytesIO(obj["Body"].read()))


# --- Widgets ---
files = list_files()
file_select = pn.widgets.Select(name="Parquet file", options=files, width=300)
x_select = pn.widgets.Select(name="X axis", options=[], width=200)
y_select = pn.widgets.Select(name="Y axis", options=[], width=200)
show_raw = pn.widgets.Checkbox(name="Show raw data", value=False)
stats_pane = pn.pane.DataFrame(width=600)
raw_pane = pn.pane.DataFrame(width=900)
plot_pane = pn.pane.Matplotlib(tight=True, sizing_mode="stretch_width")

_cache: dict = {}


def _load(filename: str) -> pl.DataFrame:
    if filename not in _cache:
        _cache[filename] = load_parquet(filename)
    return _cache[filename]


def update_columns(event):
    df = _load(file_select.value)
    cols = df.columns
    x_select.options = cols
    x_select.value = "Time" if "Time" in cols else cols[0]
    y_select.options = cols
    y_select.value = cols[1] if len(cols) > 1 else cols[0]


def update_plot(*events):
    filename = file_select.value
    col_x = x_select.value
    col_y = y_select.value
    if not filename or not col_x or not col_y:
        return

    df = _load(filename)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df[col_x].to_list(),
        df[col_y].to_list(),
        marker=".",
        linestyle="-",
        alpha=0.6,
        color="#2c3e50",
    )
    ax.set_title(f"{os.path.splitext(filename)[0]}  —  {col_y} vs {col_x}", fontsize=13)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_pane.object = fig
    plt.close(fig)

    stats_pane.object = df.select([col_x, col_y]).describe().to_pandas()

    if show_raw.value:
        raw_pane.object = df.tail(200).to_pandas()
    else:
        raw_pane.object = None


# Wire up callbacks
file_select.param.watch(update_columns, "value")
file_select.param.watch(update_plot, "value")
x_select.param.watch(update_plot, "value")
y_select.param.watch(update_plot, "value")
show_raw.param.watch(update_plot, "value")

# Initialise with first file
if files:
    update_columns(None)
    update_plot()

# --- Layout ---
sidebar = pn.Column(
    pn.pane.Markdown("## Controls"),
    file_select,
    x_select,
    y_select,
    show_raw,
    width=320,
)

main = pn.Column(
    pn.pane.Markdown("# RustFS Magnetrun Viewer"),
    plot_pane,
    pn.pane.Markdown("### Column statistics"),
    stats_pane,
    pn.pane.Markdown("### Raw data (last 200 rows)"),
    raw_pane,
)

pn.template.FastListTemplate(
    title="RustFS Magnetrun Viewer",
    sidebar=[sidebar],
    main=[main],
).servable()
