#!/bin/bash

# Change to app directory
cd /app

# Start Streamlit (dependencies are already installed in the Docker image)
streamlit run src/interfaces/web_ui.py --server.port=8501 --server.address=0.0.0.0
