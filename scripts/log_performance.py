"""
Log model performance metrics to MLflow.
This script is executed by GitHub Actions to log model performance after retraining.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
import joblib
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import project utilities
from utils.data_loader import load_fashion_mnist
from utils.feature_engineering import apply_feature_engineering

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"performance_logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_metrics(model, X, y):
    """Calculate and return performance metrics for the model."""
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
    }
    
    # Calculate per-class metrics for detailed monitoring
    class_metrics = {}
    for class_idx in range(10):  # 10 classes in Fashion MNIST
        mask = (y == class_idx)
        if np.any(mask):
            y_true_class = y[mask]
            y_pred_class = y_pred[mask]
            class_precision = precision_score(y_true_class, y_pred_class, average=None, zero_division=0)
            class_recall = recall_score(y_true_class, y_pred_class, average=None, zero_division=0)
            class_metrics[f'class_{class_idx}_precision'] = float(np.mean(class_precision))
            class_metrics[f'class_{class_idx}_recall'] = float(np.mean(class_recall))
    
    # Combine all metrics
    metrics.update(class_metrics)
    
    return metrics

def main():
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        experiment_name = f"model_performance_logging_{datetime.now().strftime('%Y%m%d')}"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Load the latest model
        model_path = os.path.join('models', 'latest_model.joblib')
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found!")
            return 1
            
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        
        # Apply feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_test_processed, _ = apply_feature_engineering(X_train, X_test)
        
        # Start MLflow run
        with mlflow.start_run(run_name="performance_logging") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Calculate metrics on test data
            logger.info("Calculating performance metrics on test data...")
            metrics = calculate_metrics(model, X_test_processed, y_test)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"Metric - {metric_name}: {metric_value}")
            
            # Log model metadata
            model_type = type(model).__name__
            mlflow.log_param("model_type", model_type)
            
            # Check for performance regression
            # This could compare against historical runs or a baseline
            perf_threshold = 0.8  # Example threshold
            if metrics['accuracy'] < perf_threshold:
                logger.warning(f"Performance regression detected! Accuracy {metrics['accuracy']} is below threshold {perf_threshold}")
                mlflow.log_param("performance_regression", True)
            else:
                logger.info(f"No performance regression detected. Accuracy: {metrics['accuracy']}")
                mlflow.log_param("performance_regression", False)
            
            # Save a report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report = {
                "timestamp": timestamp,
                "model_type": model_type,
                "metrics": metrics,
                "performance_regression": metrics['accuracy'] < perf_threshold
            }
            
            # Create the directory if it doesn't exist
            os.makedirs('reports/model_performance', exist_ok=True)
            
            report_path = os.path.join('reports/model_performance', f'performance_log_{timestamp}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log the report as an artifact
            mlflow.log_artifact(report_path)
            logger.info(f"Performance report saved to {report_path}")
        
        logger.info("Performance logging completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during performance logging: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())