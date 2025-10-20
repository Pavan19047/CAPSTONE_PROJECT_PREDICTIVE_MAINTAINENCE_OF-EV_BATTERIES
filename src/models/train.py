"""
EV Battery Digital Twin - ML Model Training Pipeline
Trains XGBoost models for RUL prediction and failure classification
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatteryMLTrainer:
    """Train ML models for battery RUL prediction and failure classification"""
    
    def __init__(self, dataset_path: str, models_dir: str = "models"):
        """
        Initialize the trainer
        
        Args:
            dataset_path: Path to the EV dataset CSV
            models_dir: Directory to save trained models
        """
        self.dataset_path = dataset_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Feature columns
        self.feature_cols = [
            'SoC', 'SoH', 'Battery_Voltage', 'Battery_Current',
            'Battery_Temperature', 'Charge_Cycles', 'Power_Consumption'
        ]
        
        # Target columns
        self.rul_target = 'RUL'
        self.failure_target = 'Failure_Probability'
        
        # Models
        self.rul_model = None
        self.failure_model = None
        self.scaler = None
        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("EV_Battery_Digital_Twin")
        
        logger.info(f"✅ Trainer initialized with dataset: {dataset_path}")
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the dataset
        
        Returns:
            X: Features dataframe
            y_rul: RUL target
            y_failure: Failure probability target
        """
        logger.info("📊 Loading dataset...")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values
        logger.info("🔧 Preprocessing data...")
        
        # Forward fill then backward fill for any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining rows with NaN in critical columns
        critical_cols = self.feature_cols + [self.rul_target, self.failure_target]
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        logger.info(f"After handling missing values: {df.shape}")
        
        # Remove outliers using IQR method for feature columns only
        logger.info("🔍 Removing outliers...")
        for col in self.feature_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"After outlier removal: {df.shape}")
        
        # Extract features and targets
        X = df[self.feature_cols].copy()
        y_rul = df[self.rul_target].copy()
        
        # For failure probability: the dataset has values 0 or 1 already
        # Convert to binary if needed
        if self.failure_target in df.columns:
            # If Failure_Probability is already 0/1, use as is; if 0.0-1.0, threshold at 0.5
            failure_vals = df[self.failure_target].unique()
            if set(failure_vals).issubset({0, 1}):
                y_failure = df[self.failure_target].astype(int)
            else:
                y_failure = (df[self.failure_target] > 0.5).astype(int)
        else:
            # Fallback: Create synthetic failure labels based on RUL
            y_failure = (y_rul < 100).astype(int)
        
        logger.info(f"✅ Data loaded: X={X.shape}, y_rul={y_rul.shape}, y_failure={y_failure.shape}")
        logger.info(f"Feature columns: {X.columns.tolist()}")
        logger.info(f"RUL range: {y_rul.min():.2f} - {y_rul.max():.2f}")
        logger.info(f"RUL statistics - Mean: {y_rul.mean():.2f}, Std: {y_rul.std():.2f}")
        logger.info(f"Failure class distribution: {y_failure.value_counts().to_dict()}")
        logger.info(f"Failure class balance: {(y_failure.sum() / len(y_failure) * 100):.2f}% positive class")
        
        return X, y_rul, y_failure
    
    def train_rul_model(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Train XGBoost model for RUL prediction
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("🎯 Training RUL Prediction Model...")
        
        with mlflow.start_run(run_name="RUL_Prediction"):
            # Define hyperparameters - optimized for larger dataset
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
                'tree_method': 'hist'  # Faster for large datasets
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            self.rul_model = XGBRegressor(**params)
            self.rul_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred_train = self.rul_model.predict(X_train)
            y_pred_test = self.rul_model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Feature importance
            feature_importance = dict(zip(X_train.columns, self.rul_model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance_rul.json")
            
            # Log model parameters only (skip model artifact due to MLflow version)
            mlflow.log_params({
                "n_estimators": self.rul_model.n_estimators,
                "learning_rate": self.rul_model.learning_rate,
                "max_depth": self.rul_model.max_depth
            })
            
            logger.info(f"✅ RUL Model - R² Score: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.2f}")
            
            return metrics
    
    def train_failure_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Train XGBoost model for failure classification
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("⚠️ Training Failure Classification Model...")
        
        with mlflow.start_run(run_name="Failure_Classification"):
            # Define hyperparameters - optimized for larger dataset
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),  # Handle class imbalance
                'random_state': 42,
                'n_jobs': -1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist'  # Faster for large datasets
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            self.failure_model = XGBClassifier(**params)
            self.failure_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred_train = self.failure_model.predict(X_train)
            y_pred_test = self.failure_model.predict(X_test)
            y_pred_proba_test = self.failure_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
                'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
                'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
                'test_auc': roc_auc_score(y_test, y_pred_proba_test)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Feature importance
            feature_importance = dict(zip(X_train.columns, self.failure_model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance_failure.json")
            
            # Log model parameters only (skip model artifact due to MLflow version)
            mlflow.log_params({
                "n_estimators": self.failure_model.n_estimators,
                "learning_rate": self.failure_model.learning_rate,
                "max_depth": self.failure_model.max_depth
            })
            
            logger.info(f"✅ Failure Model - Accuracy: {metrics['test_accuracy']:.4f}, AUC: {metrics['test_auc']:.4f}")
            
            return metrics
    
    def save_models(self):
        """Save trained models and scaler to disk"""
        logger.info("💾 Saving models to disk...")
        
        # Save RUL model
        rul_model_path = self.models_dir / "rul_model.pkl"
        joblib.dump(self.rul_model, rul_model_path)
        logger.info(f"✅ RUL model saved: {rul_model_path}")
        
        # Save failure model
        failure_model_path = self.models_dir / "failure_model.pkl"
        joblib.dump(self.failure_model, failure_model_path)
        logger.info(f"✅ Failure model saved: {failure_model_path}")
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler saved: {scaler_path}")
    
    def train_pipeline(self):
        """Execute complete training pipeline"""
        logger.info("🚀 Starting ML Training Pipeline...")
        
        try:
            # Load data
            X, y_rul, y_failure = self.load_and_preprocess_data()
            
            # Split data
            logger.info("✂️ Splitting data (80/20 train/test)...")
            X_train, X_test, y_rul_train, y_rul_test = train_test_split(
                X, y_rul, test_size=0.2, random_state=42
            )
            _, _, y_failure_train, y_failure_test = train_test_split(
                X, y_failure, test_size=0.2, random_state=42
            )
            
            # Feature scaling
            logger.info("📏 Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Train RUL model
            rul_metrics = self.train_rul_model(
                X_train_scaled, X_test_scaled,
                y_rul_train, y_rul_test
            )
            
            # Train failure model
            failure_metrics = self.train_failure_model(
                X_train_scaled, X_test_scaled,
                y_failure_train, y_failure_test
            )
            
            # Save models
            self.save_models()
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("🎉 TRAINING COMPLETE - MODEL PERFORMANCE SUMMARY")
            logger.info("="*60)
            logger.info("\n📊 RUL Prediction Model:")
            logger.info(f"   R² Score:  {rul_metrics['test_r2']:.4f} {'✅' if rul_metrics['test_r2'] > 0.90 else '⚠️'}")
            logger.info(f"   RMSE:      {rul_metrics['test_rmse']:.2f} cycles")
            logger.info(f"   MAE:       {rul_metrics['test_mae']:.2f} cycles")
            
            logger.info("\n⚠️ Failure Classification Model:")
            logger.info(f"   Accuracy:  {failure_metrics['test_accuracy']:.4f}")
            logger.info(f"   Precision: {failure_metrics['test_precision']:.4f}")
            logger.info(f"   Recall:    {failure_metrics['test_recall']:.4f}")
            logger.info(f"   F1 Score:  {failure_metrics['test_f1']:.4f}")
            logger.info(f"   AUC-ROC:   {failure_metrics['test_auc']:.4f}")
            
            logger.info("\n✅ Models saved to: {}".format(self.models_dir.absolute()))
            logger.info("✅ MLflow tracking: http://localhost:5000")
            logger.info("="*60 + "\n")
            
            return rul_metrics, failure_metrics
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point"""
    # Dataset path
    dataset_path = os.path.join("datasets", "EV_Predictive_Maintenance_Dataset_15min.csv")
    
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset not found: {dataset_path}")
        logger.error("Please download the dataset from Kaggle and place it in the datasets/ directory")
        sys.exit(1)
    
    # Initialize trainer
    trainer = BatteryMLTrainer(dataset_path=dataset_path)
    
    # Run training pipeline
    trainer.train_pipeline()


if __name__ == "__main__":
    main()
