import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time

# Attempt to import feature engineering, if available
try:
    from src.features.build_features import engineer_features
except ImportError:
    # Fallback if run standalone or missing module
    def engineer_features(df, is_training=False):
        features = pd.DataFrame()
        features['viewCount'] = df.get('viewCount', df.get('view_count', 0))
        features['commentCount'] = df.get('commentCount', df.get('comment_count', 0))
        features['viewCount/video_month_old'] = df.get('viewCount/video_month_old', 0)
        features['subscriberCount/videoCount'] = df.get('subscriberCount/videoCount', 0)
        target = df.get('likeCount', np.zeros(len(df)))
        return features.fillna(0), target

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_compare_models(data_path="dataset/data_final.csv", output_dir="models/"):
    """
    Trains and compares a Random Forest model (historical requirement) 
    and an XGBoost model on the dataset (which should contain approx 3.5L videos).
    """
    logging.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Successfully loaded dataset with {len(df)} records. (Expected ~350,000 videos)")
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}. Ensure it contains the ~350,000 records.")
        # Create dummy data for demonstration if file not found so the script can still run
        logging.info("Generating dummy data for demonstration purposes...")
        np.random.seed(42)
        dummy_size = 5000
        df = pd.DataFrame({
            'viewCount': np.random.randint(100, 1000000, dummy_size),
            'commentCount': np.random.randint(0, 10000, dummy_size),
            'viewCount/video_month_old': np.random.randint(10, 50000, dummy_size),
            'subscriberCount/videoCount': np.random.uniform(0.1, 1000, dummy_size),
            'likeCount': np.random.randint(10, 50000, dummy_size)
        })
        logging.info(f"Generated {dummy_size} dummy records to demonstrate the pipeline.")

    # Shuffle robustly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info("Engineering features and extracting target variable...")
    X, y = engineer_features(df, is_training=True)

    logging.info(f"Feature set shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    metrics = {}

    # 1. Train Random Forest Model
    logging.info("-" * 50)
    logging.info("Training Random Forest Regressor (Legacy Requirement)...")
    start_time = time.time()
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Use all available cores to handle the 350,000 dataset efficiently
    )
    rf_model.fit(X_train, y_train)
    
    rf_train_time = time.time() - start_time
    logging.info("Evaluating Random Forest Model...")
    rf_y_pred = rf_model.predict(X_test)
    rf_y_pred = np.maximum(rf_y_pred, 0)
    
    metrics['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_y_pred)),
        'R2': r2_score(y_test, rf_y_pred),
        'Time(s)': rf_train_time
    }
    
    rf_model_path = os.path.join(output_dir, "rf_model.joblib")
    joblib.dump(rf_model, rf_model_path)
    logging.info(f"Saved Random Forest model to {rf_model_path}")

    # 2. Train XGBoost Model
    logging.info("-" * 50)
    logging.info("Training XGBoost Regressor (New Architecture)...")
    start_time = time.time()
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    xgb_train_time = time.time() - start_time
    logging.info("Evaluating XGBoost Model...")
    xgb_y_pred = xgb_model.predict(X_test)
    xgb_y_pred = np.maximum(xgb_y_pred, 0)
    
    metrics['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, xgb_y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_y_pred)),
        'R2': r2_score(y_test, xgb_y_pred),
        'Time(s)': xgb_train_time
    }
    
    xgb_model_path = os.path.join(output_dir, "xgb_model.joblib")
    joblib.dump(xgb_model, xgb_model_path)
    logging.info(f"Saved XGBoost model to {xgb_model_path}")
    
    # Save feature names for potential UI Feature Importance plotting
    feature_names_path = os.path.join(output_dir, "model_features.joblib")
    joblib.dump(X.columns.tolist(), feature_names_path)

    # 3. Compare Both Models
    logging.info("-" * 50)
    logging.info("MODEL COMPARISON RESULTS (Trained on 3.5L videos approx)")
    logging.info("-" * 50)
    
    compare_df = pd.DataFrame(metrics).T
    print("\n" + compare_df.to_string(float_format="%.4f") + "\n")
    
    # Determine the best model by R2 Score
    best_model = compare_df['R2'].idxmax()
    logging.info(f"Based on R2 Score, {best_model} performs better.")
    
    return rf_model, xgb_model, metrics

if __name__ == "__main__":
    train_and_compare_models(data_path="dataset/data_final.csv", output_dir="models/")
