"""
Run benchmarks to compare model performance across versions.
This script is designed to be run by GitHub Actions as part of the CI/CD pipeline.
"""

import os
import sys
import glob
import logging
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import project utilities
from utils.data_loader import load_fashion_mnist, get_class_names
from utils.feature_engineering import apply_feature_engineering

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def benchmark_model(model, X, y, model_name=None):
    """Benchmark a model's performance and speed."""
    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'prediction_time_seconds': prediction_time,
        'predictions_per_second': len(X) / prediction_time if prediction_time > 0 else 0
    }
    
    if model_name:
        logger.info(f"Model {model_name} performance:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")
    
    return metrics, y_pred

def find_model_files():
    """Find all saved model files."""
    models_dir = 'models'
    model_files = glob.glob(os.path.join(models_dir, '*.joblib'))
    return model_files

def main():
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        experiment_name = f"model_benchmarking_{datetime.now().strftime('%Y%m%d')}"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Create output directories
        os.makedirs('reports/model_performance', exist_ok=True)
        
        # Load Fashion MNIST dataset
        logger.info("Loading Fashion MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        class_names = get_class_names()
        
        # Apply feature engineering
        logger.info("Engineering features...")
        X_train_processed, X_test_processed, _ = apply_feature_engineering(X_train, X_test)
        
        # Find all model files
        model_files = find_model_files()
        logger.info(f"Found {len(model_files)} model files: {model_files}")
        
        if not model_files:
            logger.warning("No model files found to benchmark!")
            return 1
        
        # Start MLflow run
        with mlflow.start_run(run_name="model_benchmarking") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Dictionary to store benchmark results
            benchmark_results = {}
            all_predictions = {}
            
            # Benchmark each model
            for model_file in model_files:
                model_name = os.path.basename(model_file)
                logger.info(f"Benchmarking model: {model_name}")
                
                try:
                    # Load the model
                    model = joblib.load(model_file)
                    
                    # Benchmark on test data
                    metrics, y_pred = benchmark_model(model, X_test_processed, y_test, model_name)
                    
                    # Store results
                    benchmark_results[model_name] = metrics
                    all_predictions[model_name] = y_pred.tolist()  # Convert to list for JSON serialization
                    
                    # Log to MLflow
                    with mlflow.start_run(run_name=f"benchmark_{model_name}", nested=True):
                        # Log model info
                        mlflow.log_param("model_file", model_name)
                        mlflow.log_param("model_type", type(model).__name__)
                        
                        # Log metrics
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        
                        # Generate and log confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cm, interpolation='nearest', cmap='Blues')
                        plt.title(f'Confusion Matrix - {model_name}')
                        plt.colorbar()
                        tick_marks = np.arange(len(class_names))
                        plt.xticks(tick_marks, class_names, rotation=45)
                        plt.yticks(tick_marks, class_names)
                        
                        # Add text annotations
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                plt.text(j, i, format(cm[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black")
                        
                        plt.xlabel('Predicted label')
                        plt.ylabel('True label')
                        plt.tight_layout()
                        
                        # Save and log confusion matrix
                        cm_path = f'reports/model_performance/confusion_matrix_{model_name}.png'
                        plt.savefig(cm_path)
                        mlflow.log_artifact(cm_path)
                    
                except Exception as e:
                    logger.error(f"Error benchmarking model {model_name}: {str(e)}")
            
            # Compare models performance
            if benchmark_results:
                # Create comparison visualizations
                plt.figure(figsize=(12, 6))
                
                # Plot accuracy comparison
                model_names = list(benchmark_results.keys())
                accuracies = [results['accuracy'] for results in benchmark_results.values()]
                
                y_pos = np.arange(len(model_names))
                plt.bar(y_pos, accuracies, align='center', alpha=0.7)
                plt.xticks(y_pos, [name[:15] + '...' if len(name) > 15 else name for name in model_names], rotation=45)
                plt.ylabel('Accuracy')
                plt.title('Model Accuracy Comparison')
                plt.ylim(0, 1.0)
                
                # Add value labels on bars
                for i, v in enumerate(accuracies):
                    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
                
                plt.tight_layout()
                accuracy_plot_path = 'reports/model_performance/accuracy_comparison.png'
                plt.savefig(accuracy_plot_path)
                mlflow.log_artifact(accuracy_plot_path)
                
                # Plot prediction speed comparison
                plt.figure(figsize=(12, 6))
                pred_speeds = [results['predictions_per_second'] for results in benchmark_results.values()]
                
                plt.bar(y_pos, pred_speeds, align='center', alpha=0.7)
                plt.xticks(y_pos, [name[:15] + '...' if len(name) > 15 else name for name in model_names], rotation=45)
                plt.ylabel('Predictions per second')
                plt.title('Model Speed Comparison')
                
                # Add value labels on bars
                for i, v in enumerate(pred_speeds):
                    plt.text(i, v + max(pred_speeds) * 0.01, f"{v:.1f}", ha='center')
                
                plt.tight_layout()
                speed_plot_path = 'reports/model_performance/speed_comparison.png'
                plt.savefig(speed_plot_path)
                mlflow.log_artifact(speed_plot_path)
                
                # Save benchmark results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_dict = {
                    "timestamp": timestamp,
                    "benchmark_results": benchmark_results
                }
                
                results_path = f'reports/model_performance/benchmark_results_{timestamp}.json'
                with open(results_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                
                mlflow.log_artifact(results_path)
                logger.info(f"Benchmark results saved to {results_path}")
                
                # Find and log the best model based on accuracy
                best_model_name = max(benchmark_results.items(), key=lambda x: x[1]['accuracy'])[0]
                logger.info(f"Best performing model: {best_model_name}")
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_model_accuracy", benchmark_results[best_model_name]['accuracy'])
            
        logger.info("Model benchmarking completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during model benchmarking: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())