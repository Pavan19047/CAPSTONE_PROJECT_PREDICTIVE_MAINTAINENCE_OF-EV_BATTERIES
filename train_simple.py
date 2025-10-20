"""
Simplified Training Script - No MLflow Dependencies
Trains XGBoost models for RUL prediction and failure classification
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import xgboost as xgb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("datasets/EV_Predictive_Maintenance_Dataset_15min.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    logger.info("📊 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset shape: {df.shape}")
    
    logger.info("🔧 Preprocessing data...")
    
    # Handle missing values
    df = df.ffill().bfill()
    logger.info(f"After handling missing values: {df.shape}")
    
    # Remove outliers using IQR method
    logger.info("🔍 Removing outliers...")
    feature_cols = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Current', 
                    'Battery_Temperature', 'Charge_Cycles', 'Power_Consumption']
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    logger.info(f"After outlier removal: {df.shape}")
    
    # Extract features and targets
    X = df[feature_cols]
    y_rul = df['RUL']
    y_failure = df['Failure_Probability']
    
    # Convert failure probabilities to binary (if needed)
    if y_failure.dtype == object or y_failure.max() > 1:
        y_failure = (y_failure > 0.5).astype(int)
    else:
        y_failure = y_failure.astype(int)
    
    logger.info(f"✅ Data loaded: X={X.shape}, y_rul={y_rul.shape}, y_failure={y_failure.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    logger.info(f"RUL range: {y_rul.min():.2f} - {y_rul.max():.2f}")
    logger.info(f"RUL statistics - Mean: {y_rul.mean():.2f}, Std: {y_rul.std():.2f}")
    logger.info(f"Failure class distribution: {dict(y_failure.value_counts())}")
    logger.info(f"Failure class balance: {(y_failure.sum() / len(y_failure) * 100):.2f}% positive class")
    
    return X, y_rul, y_failure

def train_rul_model(X_train, X_test, y_train, y_test):
    """Train RUL prediction model"""
    logger.info("🎯 Training RUL Prediction Model...")
    
    params = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    logger.info(f"✅ RUL Model - R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}")
    
    return model, metrics

def train_failure_model(X_train, X_test, y_train, y_test):
    """Train failure classification model"""
    logger.info("⚠️ Training Failure Classification Model...")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
        'test_auc': roc_auc_score(y_test, y_pred_proba_test)
    }
    
    logger.info(f"✅ Failure Model - Accuracy: {metrics['test_accuracy']:.4f}, F1: {metrics['test_f1']:.4f}, AUC: {metrics['test_auc']:.4f}")
    
    return model, metrics

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting ML Training Pipeline...")
    
    # Load data
    X, y_rul, y_failure = load_and_preprocess_data()
    
    # Split data
    logger.info("✂️ Splitting data (80/20 train/test)...")
    X_train, X_test, y_rul_train, y_rul_test, y_fail_train, y_fail_test = train_test_split(
        X, y_rul, y_failure, test_size=0.2, random_state=42
    )
    
    # Scale features
    logger.info("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Train RUL model
    rul_model, rul_metrics = train_rul_model(X_train_scaled, X_test_scaled, y_rul_train, y_rul_test)
    
    # Train failure model
    failure_model, failure_metrics = train_failure_model(X_train_scaled, X_test_scaled, y_fail_train, y_fail_test)
    
    # Save models
    logger.info("💾 Saving models...")
    joblib.dump(rul_model, MODELS_DIR / "rul_model.pkl")
    joblib.dump(failure_model, MODELS_DIR / "failure_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    logger.info(f"✅ Models saved to {MODELS_DIR}/")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print("\n🎯 RUL Prediction Model:")
    for k, v in rul_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    print("\n⚠️ Failure Classification Model:")
    for k, v in failure_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    print("\n💾 Saved Files:")
    print(f"  - {MODELS_DIR}/rul_model.pkl")
    print(f"  - {MODELS_DIR}/failure_model.pkl")
    print(f"  - {MODELS_DIR}/scaler.pkl")
    print("="*60)
    
    logger.info("✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
