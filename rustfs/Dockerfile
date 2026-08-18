# Use a slim Python image to keep it fast and small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Polars/Boto3 edge cases)
# libgl1-mesa-glx, libglib2.0-0, python3-tk are required for X11 display forwarding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your test scripts into the container
COPY . .

# Keep the container alive so you can exec into it and run tests
CMD ["tail", "-f", "/dev/null"]

