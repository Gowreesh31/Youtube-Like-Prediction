# App environment
FROM python:3.10-slim

# Working Directory
WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# Install Dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Project
COPY . /app

# Expose Streamlit Port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Application
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
