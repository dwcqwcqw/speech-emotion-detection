#!/usr/bin/env python3

import os
import sys
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run Streamlit app with subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"])
