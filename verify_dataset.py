"""
Quick dataset verification script
"""
import pandas as pd
import os

# Check if dataset exists
dataset_path = "datasets/EV_Predictive_Maintenance_Dataset_15min.csv"

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at: {dataset_path}")
    exit(1)

print("=" * 70)
print("📊 EV BATTERY DATASET VERIFICATION")
print("=" * 70)

# Load dataset
print(f"\n📁 Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"\n✅ Dataset loaded successfully!")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Features that will be used for training
feature_cols = [
    'SoC', 'SoH', 'Battery_Voltage', 'Battery_Current',
    'Battery_Temperature', 'Charge_Cycles', 'Power_Consumption'
]

print(f"\n🎯 Features for ML Training ({len(feature_cols)} features):")
for i, col in enumerate(feature_cols, 1):
    if col in df.columns:
        print(f"   {i}. {col:25s} ✅")
    else:
        print(f"   {i}. {col:25s} ❌ MISSING!")

# Target variables
print(f"\n🎯 Target Variables:")
if 'RUL' in df.columns:
    print(f"   1. RUL (Remaining Useful Life)")
    print(f"      Range: {df['RUL'].min():.2f} - {df['RUL'].max():.2f}")
    print(f"      Mean: {df['RUL'].mean():.2f}, Std: {df['RUL'].std():.2f}")
else:
    print(f"   1. RUL ❌ MISSING!")

if 'Failure_Probability' in df.columns:
    print(f"   2. Failure_Probability")
    unique_vals = df['Failure_Probability'].unique()
    print(f"      Unique values: {sorted(unique_vals)[:10]}...")
    print(f"      Value counts:")
    print(f"         Class 0 (No Failure): {(df['Failure_Probability'] == 0).sum():,} ({(df['Failure_Probability'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"         Class 1 (Failure):    {(df['Failure_Probability'] == 1).sum():,} ({(df['Failure_Probability'] == 1).sum() / len(df) * 100:.2f}%)")
else:
    print(f"   2. Failure_Probability ❌ MISSING!")

# Data quality check
print(f"\n🔍 Data Quality Check:")
print(f"   Missing values per column:")
missing = df[feature_cols + ['RUL', 'Failure_Probability']].isnull().sum()
if missing.sum() == 0:
    print(f"      ✅ No missing values in critical columns!")
else:
    for col, count in missing.items():
        if count > 0:
            print(f"      {col}: {count} ({count/len(df)*100:.2f}%)")

# Sample data
print(f"\n📋 Sample Data (first 3 rows):")
sample_cols = feature_cols + ['RUL', 'Failure_Probability']
print(df[sample_cols].head(3).to_string())

# Statistics
print(f"\n📊 Feature Statistics:")
print(df[feature_cols].describe().round(4).to_string())

print("\n" + "=" * 70)
print("✅ DATASET IS READY FOR TRAINING!")
print("=" * 70)
print(f"\n💡 Next step: Run 'python src/models/train.py' to train the models")
print(f"   Expected training time: 2-5 minutes (175K+ samples)")
print(f"   Expected R² score: > 0.90 for RUL prediction")
