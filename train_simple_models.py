"""
Train simple CPU-based ML models for battery predictions
This replaces the GPU-based cuML models with scikit-learn models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Load dataset
print("Loading dataset...")
df = pd.read_csv('datasets/EV_Predictive_Maintenance_Dataset_15min.csv')

# Define features and targets
feature_cols = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Current', 
                'Battery_Temperature', 'Charge_Cycles', 'Power_Consumption']

targets = {
    'SoH': 'SoH',
    'Battery_Temperature': 'Battery_Temperature',
    'RUL': 'RUL',
    'Failure_Probability': 'Failure_Probability'
}

# Prepare data
X = df[feature_cols].fillna(df[feature_cols].mean())

# Train models
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

print("\nTraining models...")
for model_name, target_col in targets.items():
    print(f"\n📊 Training {model_name} model...")
    
    y = df[target_col].fillna(df[target_col].mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  ✅ Train R²: {train_score:.4f}")
    print(f"  ✅ Test R²: {test_score:.4f}")
    
    # Save model
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"  💾 Saved to {model_path}")

print("\n🎉 All models trained successfully!")
print(f"📁 Models saved in: {models_dir.absolute()}")
print("\n🔄 Restart the Flask app to load the new models:")
print("   python app_advanced.py")
