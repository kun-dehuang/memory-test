# Railway deployment Dockerfile for Memory Evaluation Tool

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libexif-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY ./backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create photos and identity directories
RUN mkdir -p photos identity

# Expose port
EXPOSE 8000

# Run application (use PORT env var if set, default to 8000)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
