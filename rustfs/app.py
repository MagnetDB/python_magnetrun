import boto3
import config
import polars as pl
import streamlit as st
from botocore.config import Config


## --- STEP 1: Setup Secure Connection ---
def get_s3_client():
    # We add the Cloudflare Service Token to the headers
    # This lets us pass through the "Zero Trust" firewall
    session = boto3.Session(
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY
    )
    
    return session.client(
        's3',
        endpoint_url=config.NAS_S3_URL,
        config=Config(proxies={}), # Ensure no system proxy interferes
        # Add the Cloudflare Headers here!
        extra_headers={
            "CF-Access-Client-Id": config.CF_CLIENT_ID,
            "CF-Access-Client-Secret": config.CF_CLIENT_SECRET
        }
    )

## --- STEP 2: Fetch Data Logic ---
@st.cache_data(ttl=300) # Cache for 5 mins to save bandwidth
def load_nas_data(bucket_name, file_key):
    s3 = get_s3_client()
    try:
        # Download file to a memory buffer
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read()
        
        # Parse with Polars (very fast)
        return pl.read_csv(content)
    except (OSError, ValueError, RuntimeError) as e:
        st.error(f"Security/Connection Error: {e}")
        return None

## --- STEP 3: The Dashboard UI ---
st.set_page_config(page_title="NAS Remote Dashboard", layout="wide")
st.title("📈 Real-time NAS Data Analytics")

df = load_nas_data("raw-logs", "sensor_data.csv")

if df is not None:
    st.success("Successfully connected to NAS via Secure Tunnel")
    
    # Simple Visualization
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.line_chart(df.to_pandas().set_index("timestamp"))
    
    with col2:
        st.write("Latest Raw Entries")
        st.dataframe(df.tail(10).to_pandas())

