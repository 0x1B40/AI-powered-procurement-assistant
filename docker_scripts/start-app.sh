#!/bin/bash

# Change to app directory
cd /app

# Start the application (dependencies are already installed in the Docker image)
uvicorn src.interfaces.api_server:app --host 0.0.0.0 --port 8000
