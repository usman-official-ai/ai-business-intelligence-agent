"""
AI Business Intelligence Agent - Streamlit App
Main entry point for deployment
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
import streamlit as st
from dashboard.app import *

# Set page config
st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)

# Run the main app
if __name__ == "__main__":
    pass