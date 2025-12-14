"""
Check if required files exist for Railway deployment
"""
import os

print("=" * 60)
print("Checking required files for Loan Prediction System...")
print("=" * 60)

required_files = {
    'UI File': 'loan_prediction_ui.py',
    'Data File': 'Comfort.csv',
    'Model Files': [
        'models/loan_prediction_model.pkl',
        'models/label_encoders.pkl',
        'models/scaler.pkl',
        'models/feature_names.pkl'
    ]
}

all_good = True

# Check UI file
if os.path.exists(required_files['UI File']):
    print(f"✅ {required_files['UI File']} found")
else:
    print(f"❌ {required_files['UI File']} NOT FOUND")
    all_good = False

# Check data file
if os.path.exists(required_files['Data File']):
    print(f"✅ {required_files['Data File']} found")
else:
    print(f"⚠️  {required_files['Data File']} NOT FOUND (optional for predictions)")

# Check model files
print("\nModel Files:")
model_found = False
for model_file in required_files['Model Files']:
    if os.path.exists(model_file):
        print(f"✅ {model_file} found")
        model_found = True
    else:
        print(f"❌ {model_file} NOT FOUND")

if not model_found:
    print("\n⚠️  WARNING: No model files found!")
    print("Please train the model using Complete_Loan_Prediction_Project.ipynb")
    print("and commit the models/ directory to your repository.")
    all_good = False

print("=" * 60)
if all_good:
    print("✅ All required files present. App should work!")
else:
    print("❌ Some files are missing. App may not work correctly.")
print("=" * 60)