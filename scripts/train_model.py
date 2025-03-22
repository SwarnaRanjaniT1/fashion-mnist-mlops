#!/usr/bin/env python3
"""
Train a model on the Fashion MNIST dataset and save it to the models directory.
This script is run by GitHub Actions during the CI/CD pipeline.
"""

import os
import sys
import mlflow
import mlflow.sklearn
import pickle
import joblib
import json
import time

# Add the root directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_fashion_mnist, preprocess_data
from utils.feature_engineering import apply_feature_engineering
from utils.model_selection import select_model_with_automl
from utils.hyperparameter_optimization import optimize_hyperparameters

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Configure MLflow
mlflow.set_tracking_uri('file:./mlruns')
experiment_name = f"fashion_mnist_training_{int(time.time())}"
mlflow.set_experiment(experiment_name)

def main():
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    
    print("Preprocessing data...")
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    print("Applying feature engineering...")
    X_train_processed, X_test_processed, feature_pipeline = apply_feature_engineering(
        X_train, X_test
    )
    
    print("Selecting best model with AutoML...")
    best_model, all_models = select_model_with_automl(
        X_train_processed, y_train, X_test_processed, y_test, n_trials=5
    )
    
    print("Optimizing hyperparameters...")
    optimized_model, optimization_history = optimize_hyperparameters(
        best_model, X_train_processed, y_train, X_test_processed, y_test
    )
    
    print("Saving model artifacts...")
    # Save the optimized model
    model_path = os.path.join("models", "fashion_mnist_model.joblib")
    joblib.dump(optimized_model, model_path)
    
    # Save the feature pipeline
    pipeline_path = os.path.join("models", "feature_pipeline.joblib")
    joblib.dump(feature_pipeline, pipeline_path)
    
    # Save model metadata
    metadata = {
        "model_type": type(optimized_model).__name__,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_count": X_train_processed.shape[1],
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0]
    }
    
    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model training completed successfully, artifacts saved to models/ directory")
    
    # Log model with MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(optimized_model, "model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(pipeline_path)
        mlflow.log_artifact(metadata_path)
        
        # Log metrics
        from sklearn.metrics import accuracy_score
        y_pred = optimized_model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", accuracy)
        
        print(f"Model logged to MLflow with run_id: {run.info.run_id}")
        print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()