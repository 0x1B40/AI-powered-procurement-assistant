# Multi-stage Dockerfile for Procurement Assistant
# Build stage: Install dependencies
FROM python:3.11-slim as builder

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Runtime stage: Copy dependencies and application code
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime system dependencies (if any)
RUN apt-get update && apt-get install -y \
    # Add any runtime dependencies here if needed
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY requirements.txt .
COPY docker_scripts/ ./docker_scripts/
COPY data_scripts/ ./data_scripts/

# Create data directory
RUN mkdir -p /app/data

# Make scripts executable
RUN chmod +x /app/docker_scripts/start-app.sh /app/docker_scripts/start-streamlit.sh

# Expose ports
EXPOSE 8000 8501

# Default command (will be overridden by docker-compose)
CMD ["bash"]
