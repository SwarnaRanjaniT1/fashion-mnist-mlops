"""
Detect drift in the Fashion MNIST dataset and generate drift reports.
This script is run by GitHub Actions during the CI/CD pipeline.
"""

import os
import sys
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import project utilities
from utils.data_loader import load_fashion_mnist
from utils.feature_engineering import apply_feature_engineering
from utils.drift_detection import detect_drift, monitor_drift_over_time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"drift_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run drift evaluation and log results to MLflow."""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        experiment_name = f"drift_evaluation_{datetime.now().strftime('%Y%m%d')}"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Create output directories
        os.makedirs('reports/drift_detection', exist_ok=True)
        
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        
        # Apply feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_test_processed, _ = apply_feature_engineering(X_train, X_test)
        
        # Find the latest model
        models_dir = 'models'
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        if not model_files:
            logger.error("No model files found for drift evaluation!")
            return 1
        
        # Try to use latest_model.joblib if it exists
        if 'latest_model.joblib' in model_files:
            model_path = os.path.join(models_dir, 'latest_model.joblib')
        else:
            # Otherwise, use the alphabetically last model file (which would be the latest by timestamp)
            model_path = os.path.join(models_dir, sorted(model_files)[-1])
        
        logger.info(f"Using model: {model_path}")
        model = joblib.load(model_path)
        
        # Start MLflow run
        with mlflow.start_run(run_name="drift_evaluation") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Log model info
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("model_type", type(model).__name__)
            
            # Detect drift with different intensities
            logger.info("Detecting drift with various intensities...")
            drift_intensities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            drift_results = {}
            
            for intensity in drift_intensities:
                logger.info(f"Evaluating drift with intensity {intensity}...")
                result = detect_drift(model, X_test_processed, y_test, drift_intensity=intensity)
                drift_results[str(intensity)] = {
                    'accuracy': result['accuracy'],
                    'drift_score': result['drift_score'],
                    'drift_detected': result['drift_detected']
                }
                
                # Log to MLflow
                mlflow.log_metric(f"accuracy_drift_{intensity}", result['accuracy'])
                mlflow.log_metric(f"drift_score_{intensity}", result['drift_score'])
            
            # Monitor drift over time
            logger.info("Monitoring drift over time...")
            drift_monitoring = monitor_drift_over_time(
                model, X_test_processed, y_test, 
                n_batches=10,
                drift_progression=[i/10 for i in range(11)]  # 0.0 to 1.0 in steps of 0.1
            )
            
            # Create drift visualization
            plt.figure(figsize=(12, 6))
            
            # Plot accuracy vs drift intensity
            intensities = list(map(float, drift_results.keys()))
            accuracies = [result['accuracy'] for result in drift_results.values()]
            drift_scores = [result['drift_score'] for result in drift_results.values()]
            
            plt.plot(intensities, accuracies, 'o-', label='Accuracy', color='blue')
            plt.plot(intensities, drift_scores, 'o-', label='Drift Score', color='red')
            
            plt.xlabel('Drift Intensity')
            plt.ylabel('Score')
            plt.title('Impact of Data Drift on Model Performance')
            plt.legend()
            plt.grid(True)
            
            drift_impact_path = 'reports/drift_detection/drift_impact.png'
            plt.savefig(drift_impact_path)
            mlflow.log_artifact(drift_impact_path)
            
            # Create temporal drift visualization
            plt.figure(figsize=(12, 6))
            
            # Format timestamps for better display
            timestamps = [f"Batch {i+1}" for i in range(len(drift_monitoring['timestamps']))]
            
            plt.plot(timestamps, drift_monitoring['accuracy'], 'o-', label='Accuracy', color='blue')
            plt.plot(timestamps, drift_monitoring['drift_scores'], 'o-', label='Drift Score', color='red')
            
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.title('Model Performance Over Time with Increasing Drift')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            temporal_drift_path = 'reports/drift_detection/temporal_drift.png'
            plt.savefig(temporal_drift_path)
            mlflow.log_artifact(temporal_drift_path)
            
            # Calculate drift sensitivity
            # The rate at which accuracy decreases as drift increases
            if len(intensities) > 1 and intensities[-1] > intensities[0]:
                drift_sensitivity = (accuracies[0] - accuracies[-1]) / (intensities[-1] - intensities[0])
                mlflow.log_metric("drift_sensitivity", drift_sensitivity)
                logger.info(f"Drift sensitivity: {drift_sensitivity:.4f}")
            
            # Save drift results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dict = {
                "timestamp": timestamp,
                "model": os.path.basename(model_path),
                "drift_results": drift_results,
                "drift_monitoring": {
                    "timestamps": drift_monitoring['timestamps'],
                    "accuracies": drift_monitoring['accuracy'],
                    "drift_scores": drift_monitoring['drift_scores'],
                    "drift_intensities": drift_monitoring['drift_intensities']
                }
            }
            
            results_path = f'reports/drift_detection/drift_evaluation_{timestamp}.json'
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            mlflow.log_artifact(results_path)
            logger.info(f"Drift evaluation results saved to {results_path}")
            
            # Log drift threshold recommendations
            # Find the drift intensity where accuracy drops by more than 5%
            baseline_accuracy = accuracies[0]  # Accuracy with no drift
            for i, intensity in enumerate(intensities):
                if accuracies[i] < baseline_accuracy * 0.95:  # 5% drop
                    recommended_threshold = intensity
                    mlflow.log_param("recommended_drift_threshold", recommended_threshold)
                    logger.info(f"Recommended drift threshold: {recommended_threshold} (5% accuracy drop)")
                    break
            
        logger.info("Drift evaluation completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during drift evaluation: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())