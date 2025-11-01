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

import json
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
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
        
        # Focus on most relevant features for RUL prediction
        self.feature_cols = [
            'SoH',                # State of Health - directly related to RUL
            'Battery_Temperature', # Temperature affects degradation
            'Charge_Cycles',      # Cycle count is crucial for RUL
            'Battery_Voltage',    # Voltage patterns indicate health
            'Battery_Current',    # Current patterns indicate usage
            'Power_Consumption',  # Power usage affects degradation
            'Component_Health_Score'  # Overall health indicator
        ]
        
        # Target columns
        self.rul_target = 'RUL'
        self.failure_target = 'Failure_Probability'
        
        # Models
        self.rul_model = None
        self.failure_model = None
        self.scaler = None
        
        # Setup complete
        
        logger.info(f"✅ Trainer initialized with dataset: {dataset_path}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features focused on RUL prediction using battery degradation patterns
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with additional engineered features
        """
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Normalize key features to 0-1 range for consistent processing
        df['SoH_Normalized'] = df['SoH'] / 100 if df['SoH'].max() > 1 else df['SoH']
        df['CHS_Normalized'] = df['Component_Health_Score'] / 100 if df['Component_Health_Score'].max() > 1 else df['Component_Health_Score']
        
        # Calculate rate of change (degradation rates)
        for col in ['SoH_Normalized', 'CHS_Normalized']:
            df[f'{col}_Change'] = df[col].diff()
            df[f'{col}_Change_Rate'] = df[f'{col}_Change'] / df['Charge_Cycles'].diff()
        
        # Battery temperature features
        df['Temp_vs_Ambient'] = df['Battery_Temperature'] - df['Ambient_Temperature']
        df['Overheating_Time'] = (df['Battery_Temperature'] > 45).astype(int).rolling(window=50).sum()
        
        # Cycle-based degradation
        df['Cycles_Squared'] = df['Charge_Cycles'] ** 2
        df['Cycle_Temperature_Impact'] = df['Charge_Cycles'] * (df['Battery_Temperature'] / df['Battery_Temperature'].mean())
        
        # Power usage patterns
        df['Power_per_Cycle'] = df['Power_Consumption'] / df['Charge_Cycles']
        df['Discharge_Depth'] = abs(df['Battery_Current']) / df['Battery_Voltage']
        
        # Combined health indicators
        df['Overall_Health'] = df['SoH_Normalized'] * df['CHS_Normalized']
        df['Health_Cycle_Ratio'] = df['Overall_Health'] / df['Charge_Cycles']
        
        # Moving averages and trends (shorter windows for more immediate patterns)
        for col in ['SoH_Normalized', 'Overall_Health', 'Power_Consumption']:
            # Exponential moving averages
            df[f'{col}_EMA_Fast'] = df[col].ewm(span=10, adjust=False).mean()
            df[f'{col}_EMA_Slow'] = df[col].ewm(span=30, adjust=False).mean()
            
            # Trend direction
            df[f'{col}_Trend'] = (df[f'{col}_EMA_Fast'] > df[f'{col}_EMA_Slow']).astype(int)
        
        # Battery stress scores
        thermal_stress = (df['Battery_Temperature'] - df['Battery_Temperature'].mean()) / df['Battery_Temperature'].std()
        voltage_stress = (df['Battery_Voltage'] - df['Battery_Voltage'].mean()) / df['Battery_Voltage'].std()
        current_stress = (df['Battery_Current'] - df['Battery_Current'].mean()) / df['Battery_Current'].std()
        
        df['Stress_Score'] = (abs(thermal_stress) + abs(voltage_stress) + abs(current_stress)) / 3
        df['Cumulative_Stress'] = df['Stress_Score'].cumsum()
        
        # Calculate estimated RUL based on different factors
        # 1. Health-based RUL
        max_cycles = df['Charge_Cycles'].max()
        df['Health_Based_RUL'] = (df['Overall_Health'] / df['Overall_Health'].iloc[0]) * (max_cycles - df['Charge_Cycles'])
        
        # 2. Stress-based RUL
        df['Stress_Based_RUL'] = (1 - df['Cumulative_Stress']/df['Cumulative_Stress'].max()) * (max_cycles - df['Charge_Cycles'])
        
        # 3. Temperature-based RUL
        temp_impact = (45 - df['Battery_Temperature']) / 45  # Normalize around optimal temp
        df['Temp_Based_RUL'] = temp_impact * (max_cycles - df['Charge_Cycles'])
        
        # Clean up
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Select final feature set
        engineered_features = [
            'SoH_Normalized', 'CHS_Normalized',
            'SoH_Normalized_Change_Rate', 'CHS_Normalized_Change_Rate',
            'Overall_Health', 'Health_Cycle_Ratio',
            'Cycle_Temperature_Impact', 'Power_per_Cycle',
            'Stress_Score', 'Cumulative_Stress',
            'Health_Based_RUL', 'Stress_Based_RUL', 'Temp_Based_RUL'
        ]
        
        # Add original features
        for col in self.feature_cols:
            if col not in engineered_features:
                engineered_features.append(col)
        
        # Keep only the selected features
        df = df[engineered_features]
        
        return df

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
        
        # Handle missing values
        logger.info("🔧 Preprocessing data...")
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Engineer features
        logger.info("🔨 Engineering features...")
        df = self.engineer_features(df)
        
        # More sophisticated outlier handling
        logger.info("🔍 Handling outliers...")
        
        # First, handle missing values in critical columns
        critical_cols = self.feature_cols + [self.rul_target, self.failure_target]
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        # Define valid ranges for each feature (more relaxed)
        valid_ranges = {
            'SoH': (0, 100),
            'Battery_Temperature': (-40, 85),  # Extended operating range
            'Battery_Voltage': (0, 1000),      # Much wider voltage range
            'Battery_Current': (-1000, 1000),  # Much wider current range
            'Power_Consumption': (0, 100000),  # Much wider power range
            'Charge_Cycles': (0, 10000),       # Extended lifecycle
            'Component_Health_Score': (0, 100)
        }
        
        # First get the actual ranges in the data
        logger.info("\nActual data ranges:")
        for col in valid_ranges.keys():
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                logger.info(f"{col}: {min_val:.2f} to {max_val:.2f}")
        
        # Apply range-based filtering with more lenient thresholds
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                outliers = df[~df[col].between(min_val, max_val)]
                if len(outliers) > 0:
                    logger.info(f"\nOutliers in {col}: {len(outliers)} rows")
                    logger.info(f"Min: {outliers[col].min():.2f}, Max: {outliers[col].max():.2f}")
                df = df[df[col].between(min_val, max_val)]
        
        # Calculate RUL based on:
        # 1. Charge cycles (primary factor)
        # 2. State of Health (secondary factor)
        # 3. Temperature stress (tertiary factor)
        
        # Get maximum charge cycles from data for scaling
        max_cycles = df['Charge_Cycles'].max()
        design_life_cycles = 1000  # Typical Li-ion battery design life
        
        # Calculate base RUL from cycles
        cycle_based_rul = design_life_cycles - df['Charge_Cycles']
        
        # Adjust RUL based on SoH (normalized to 0-1)
        soh_normalized = df['SoH'] / 100 if df['SoH'].max() > 1 else df['SoH']
        soh_impact = soh_normalized * design_life_cycles
        
        # Calculate temperature stress
        temp_threshold = 45  # °C - above this temperature accelerates degradation
        temp_stress = np.maximum(0, (df['Battery_Temperature'] - temp_threshold) / 15)
        temp_impact = 1 - (temp_stress * 0.2)  # temperature can reduce RUL by up to 20%
        
        # Calculate final RUL
        df[self.rul_target] = (cycle_based_rul * soh_normalized * temp_impact).clip(0, design_life_cycles)
        
        # Calculate failure probability
        # Based on: low SoH, high temperature, high cycle count
        soh_risk = (100 - df['SoH'] * 100) / 100 if df['SoH'].max() <= 1 else (100 - df['SoH']) / 100
        temp_risk = np.maximum(0, (df['Battery_Temperature'] - 35) / 45)  # Temperature above 35°C increases risk
        cycle_risk = df['Charge_Cycles'] / design_life_cycles
        
        df[self.failure_target] = (
            0.4 * soh_risk +          # SoH has highest impact
            0.3 * temp_risk +         # Temperature has medium impact
            0.3 * cycle_risk          # Cycle count has medium impact
        ).clip(0, 1)
        
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
        
        # Hyperparameters optimized for RUL prediction with cycle-based focus
        params = {
            'n_estimators': 4000,            # More trees for complex degradation patterns
            'max_depth': 10,                 # Slightly reduced to prevent overfitting
            'learning_rate': 0.002,          # Lower learning rate for better convergence
            'min_child_weight': 3,           # More sensitive to small groups
            'subsample': 0.8,                # Prevent overfitting
            'colsample_bytree': 0.8,         # Prevent overfitting
            'colsample_bylevel': 0.8,        # Prevent overfitting
            'gamma': 0.2,                    # Increased to ensure more conservative splits
            'reg_alpha': 0.01,               # L1 regularization for feature selection
            'reg_lambda': 0.1,               # L2 regularization for stability
            'random_state': 42,              # Reproducibility
            'n_jobs': -1,                    # Use all cores
            'tree_method': 'hist',           # Fast histogram-based algorithm
            'grow_policy': 'lossguide',      # Build tree based on loss reduction
            'max_leaves': 128,               # Reduced to prevent overfitting
            'objective': 'reg:squarederror'  # For RUL regression
        }
            
        # Train final model with early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
            
        eval_params = params.copy()
        eval_params.update({
            'early_stopping_rounds': 50,
            'verbose': False
        })
            
        self.rul_model = XGBRegressor(**eval_params)
        self.rul_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=100
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
            
        # Save metrics
        metrics_path = self.models_dir / "rul_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Feature importance
        # Convert numpy float32 to regular float for JSON serialization
        feature_importance = {col: float(imp) for col, imp in zip(X_train.columns, self.rul_model.feature_importances_)}
        importance_path = self.models_dir / "feature_importance_rul.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=4)
            
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
        
        # Define hyperparameters - optimized for larger dataset
        params = {
            'n_estimators': 1000,  # Increased for better learning
            'max_depth': 8,
            'learning_rate': 0.01,  # Lower learning rate for better generalization
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],  # Multiple metrics for better tracking
            'tree_method': 'hist'
        }
            
        # Train final model with early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        eval_params = params.copy()
        eval_params.update({
            'early_stopping_rounds': 50,
            'verbose': False
        })
        
        self.failure_model = XGBClassifier(**eval_params)
        self.failure_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=100
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
        # Save metrics
        metrics_path = self.models_dir / "failure_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Feature importance
        # Convert numpy float32 to regular float for JSON serialization
        feature_importance = {col: float(imp) for col, imp in zip(X_train.columns, self.failure_model.feature_importances_)}
        importance_path = self.models_dir / "feature_importance_failure.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=4)
            
            logger.info(f"✅ Failure Model - Accuracy: {metrics['test_accuracy']:.4f}, AUC: {metrics['test_auc']:.4f}")
            
            return metrics
    
    def save_models(self):
        """Save trained models and scaler to disk"""
        logger.info("💾 Saving models to disk...")
        
        # Save RUL model
        rul_model_path = self.models_dir / "RUL.joblib"
        joblib.dump(self.rul_model, rul_model_path)
        logger.info(f"✅ RUL model saved: {rul_model_path}")
        
        # Save failure model
        failure_model_path = self.models_dir / "Failure_Probability.joblib"
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
            
            # Split data with larger training set
            logger.info("✂️ Splitting data (85/15 train/test)...")
            X_train, X_test, y_rul_train, y_rul_test = train_test_split(
                X, y_rul, test_size=0.15, random_state=42
            )
            _, _, y_failure_train, y_failure_test = train_test_split(
                X, y_failure, test_size=0.15, random_state=42
            )
            
            # Use RobustScaler for better handling of outliers
            logger.info("📏 Scaling features...")
            self.scaler = RobustScaler()
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
            # Log metrics files location
            logger.info(f"✅ RUL metrics saved to: {self.models_dir}/rul_metrics.json")
            logger.info(f"✅ Failure metrics saved to: {self.models_dir}/failure_metrics.json")
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
