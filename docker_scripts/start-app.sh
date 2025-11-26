#!/bin/bash

# Change to app directory
cd /app

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Start the application
uvicorn src.interfaces.api_server:app --host 0.0.0.0 --port 8000