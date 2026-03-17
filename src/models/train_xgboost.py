import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.features.build_features import engineer_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path, model_output_path="models/xgb_model.joblib"):
    """Trains an XGBoost regression model on the provided dataset."""
    logging.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}. Ensure data exists.")
        return None, None

    # Shuffle robustly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info("Engineering features and extracting target variable...")
    X, y = engineer_features(df, is_training=True)

    # Since we're predicting like count, using Log Transform on Target is often better for skewed metrics
    # But for simplicity, we map directly to target and use XGBoost which handles non-linearities well.
    logging.info(f"Feature set shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info("Training XGBoost Regressor...")
    # Best params can be tuned via Optuna, but here are strong defaults
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    logging.info("Evaluating Model...")
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0) # likes cannot be negative
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Model Metrics -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    logging.info(f"Saving model to {model_output_path}...")
    joblib.dump(model, model_output_path)
    
    # Save feature names for UI Feature Importance plotting
    feature_names_path = model_output_path.replace(".joblib", "_features.joblib")
    joblib.dump(X.columns.tolist(), feature_names_path)
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    # We will use the existing final dataset to train the new model
    train_model(data_path="dataset/data_final.csv", model_output_path="models/xgb_model.joblib")
