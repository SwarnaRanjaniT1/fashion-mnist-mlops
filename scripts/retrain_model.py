"""
Retrain model on the Fashion MNIST dataset with potential new data.
This script is executed by GitHub Actions as part of the scheduled workflow.
"""

import os
import sys
import logging
import time
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.data_loader import load_fashion_mnist
from utils.feature_engineering import apply_feature_engineering
from utils.model_selection import select_model_with_automl
from utils.hyperparameter_optimization import optimize_hyperparameters
from utils.model_monitoring import track_model_performance
import joblib
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"model_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        experiment_name = f"fashion_mnist_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports/model_performance', exist_ok=True)
        
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        logger.info(f"Dataset loaded: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        
        # Apply feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_test_processed, feature_pipeline = apply_feature_engineering(X_train, X_test)
        logger.info(f"Feature engineering complete: X_train_processed.shape={X_train_processed.shape}")
        
        # Start MLflow run
        with mlflow.start_run(run_name="model_retraining") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Run AutoML (with reduced time budget for GitHub Actions)
            logger.info("Running AutoML to find the best model...")
            best_model, all_models = select_model_with_automl(
                X_train_processed, y_train, X_test_processed, y_test,
                n_trials=3, time_budget=60  # Reduced for GitHub Actions
            )
            
            # Log best model name
            best_model_name = type(best_model).__name__
            logger.info(f"Best model found: {best_model_name}")
            mlflow.log_param("best_model_type", best_model_name)
            
            # Optimize hyperparameters
            logger.info("Optimizing hyperparameters for the best model...")
            optimized_model, _ = optimize_hyperparameters(
                best_model, X_train_processed, y_train, X_test_processed, y_test
            )
            logger.info("Hyperparameter optimization completed")
            
            # Track model performance
            logger.info("Tracking model performance...")
            _, metrics = track_model_performance(
                optimized_model, X_train_processed, y_train, X_test_processed, y_test
            )
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"Metric - {metric_name}: {metric_value}")
            
            # Save optimized model with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join('models', f'optimized_model_{timestamp}.joblib')
            joblib.dump(optimized_model, model_path)
            logger.info(f"Optimized model saved to {model_path}")
            
            # Also save as the "latest" model for easy reference
            latest_model_path = os.path.join('models', 'latest_model.joblib')
            joblib.dump(optimized_model, latest_model_path)
            logger.info(f"Latest model saved to {latest_model_path}")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(optimized_model, "model")
            
            # Save a simple performance report
            report = {
                "timestamp": timestamp,
                "model_type": best_model_name,
                "metrics": metrics,
                "dataset_size": {
                    "train": X_train.shape[0],
                    "test": X_test.shape[0]
                }
            }
            
            report_path = os.path.join('reports/model_performance', f'performance_report_{timestamp}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {report_path}")
            
        logger.info("Model retraining completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())