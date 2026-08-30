# Shared base image for both the Streamlit demo and the FastAPI backend.
# Build once, run either service by overriding CMD (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /workspace

# System deps required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# Default: run the Streamlit demo. Override in docker-compose for the API.
CMD ["streamlit", "run", "app/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
