"""
Streamlit demo — plot magnetrun Parquet data from RustFS.

Run locally:
    streamlit run app_streamlit.py

Run via Docker:
    docker compose exec tester streamlit run app_streamlit.py --server.port 8501
"""

import os
from io import BytesIO

import boto3
import matplotlib.pyplot as plt
import polars as pl
import streamlit as st
from botocore.config import Config

# --- CONFIG ---
RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("ACCESS_KEY", "test_user")
SECRET_KEY = os.getenv("SECRET_KEY", "test_password")
BUCKET_NAME = "dashboard-data"


@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


@st.cache_data(ttl=300)
def list_files():
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]
    except (OSError, RuntimeError) as e:
        st.error(f"Cannot list files: {e}")
        return []


@st.cache_data(ttl=300)
def load_parquet(filename: str) -> pl.DataFrame:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    return pl.read_parquet(BytesIO(obj["Body"].read()))


# --- UI ---
st.set_page_config(page_title="RustFS Magnetrun Viewer", layout="wide")
st.title("RustFS Magnetrun Viewer")

files = list_files()
if not files:
    st.warning("No files found in bucket. Upload a Parquet file first.")
    st.stop()

filename = st.sidebar.selectbox("Parquet file", files)
df = load_parquet(filename)

cols = df.columns
col_x = st.sidebar.selectbox("X axis", cols, index=cols.index("Time") if "Time" in cols else 0)
col_y = st.sidebar.selectbox("Y axis", cols, index=1 if len(cols) > 1 else 0)

st.sidebar.markdown("---")
show_raw = st.sidebar.checkbox("Show raw data", value=False)

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df[col_x].to_list(), df[col_y].to_list(), marker=".", linestyle="-", alpha=0.6, color="#2c3e50")
ax.set_title(f"{os.path.splitext(filename)[0]}  —  {col_y} vs {col_x}", fontsize=13)
ax.set_xlabel(col_x)
ax.set_ylabel(col_y)
ax.grid(True, linestyle="--", alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# --- Stats ---
with st.expander("Column statistics"):
    st.dataframe(df.select([col_x, col_y]).describe().to_pandas(), use_container_width=True)

# --- Raw data ---
if show_raw:
    st.subheader("Raw data (last 200 rows)")
    st.dataframe(df.tail(200).to_pandas(), use_container_width=True)
