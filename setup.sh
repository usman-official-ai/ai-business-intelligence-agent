#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y build-essential python3-dev

# Upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install pandas first (with binary)
pip install pandas --only-binary :all:

# Install other requirements
pip install streamlit numpy scikit-learn nltk joblib matplotlib seaborn plotly requests

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

echo "Setup complete!"