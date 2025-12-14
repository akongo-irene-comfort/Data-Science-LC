#!/usr/bin/env python3
"""Startup script for Railway deployment"""

import os
import subprocess
import sys

def main():
    # Get PORT from environment, Railway provides this
    port = os.environ.get('PORT', '8501')
    
    print(f"=" * 50)
    print(f"Starting Streamlit on port {port}")
    print(f"Address: 0.0.0.0:{port}")
    print(f"=" * 50)
    
    # Run streamlit with your actual file name
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'loan_prediction_ui.py',
        f'--server.port={port}',
        '--server.address=0.0.0.0',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        f'--browser.serverAddress=0.0.0.0',
        f'--browser.serverPort={port}'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()