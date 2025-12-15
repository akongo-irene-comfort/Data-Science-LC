#!/usr/bin/env python3
"""Startup script for Railway deployment"""

import os
import subprocess
import sys

def check_files():
    """Check if required files exist"""
    print("=" * 60)
    print("Checking required files...")
    print("=" * 60)
    
    if not os.path.exists('loan_prediction_ui.py'):
        print("❌ ERROR: loan_prediction_ui.py not found!")
        return False
    print("✅ loan_prediction_ui.py found")
    
    # Check for at least one model file
    model_files = [
        'models/loan_prediction_model.pkl',
        'models/random_forest_model.pkl'
    ]
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if not model_exists:
        print("⚠️  WARNING: No model files found in models/ directory")
        print("   App will load but predictions won't work until models are added")
    else:
        print("✅ Model files found")
    
    if not os.path.exists('Comfort.csv'):
        print("⚠️  WARNING: Comfort.csv not found")
        print("   Some analytics features may not work")
    else:
        print("✅ Comfort.csv found")
    
    print("=" * 60)
    return True

def main():
    # Railway provides PORT, fallback to 8000
    port = os.environ.get('PORT', '8000')
    
    print(f"=" * 60)
    print(f"🚀 Starting Loan Prediction App")
    print(f"📍 Port: {port}")
    print(f"🌐 Address: 0.0.0.0:{port}")
    print(f"=" * 60)
    
    # Check files
    if not check_files():
        print("❌ Critical files missing!")
        sys.exit(1)
    
    # Run streamlit
    cmd = [
        'streamlit', 'run', 'loan_prediction_ui.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"🎬 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()