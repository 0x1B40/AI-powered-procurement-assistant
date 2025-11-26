#!/bin/bash

# Change to app directory
cd /app

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Start Streamlit
streamlit run src/interfaces/web_ui.py --server.port=8501 --server.address=0.0.0.0
