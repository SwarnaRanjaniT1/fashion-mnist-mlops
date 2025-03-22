import time
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.data_loader import load_fashion_mnist, preprocess_data
from utils.feature_engineering import apply_feature_engineering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    """Run a model benchmark and log results to MLflow."""
    print("Loading Fashion MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    
    print("Preprocessing data...")
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    print("Applying feature engineering...")
    X_train_fe, X_test_fe, _ = apply_feature_engineering(X_train, X_test)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri('file:./mlruns')
    experiment_name = f"benchmark_{int(time.time())}"
    mlflow.set_experiment(experiment_name)
    
    # Define models to benchmark
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "RandomForest_Small": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        start_time = time.time()
        
        # Use a subset for faster benchmarking
        sample_size = min(5000, len(X_train_fe))
        model.fit(X_train_fe[:sample_size], y_train[:sample_size])
        
        train_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = model.predict(X_test_fe)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{model_name} results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Training Time: {train_time:.2f}s")
        
        # Log to MLflow
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            for param_name, param_value in model.get_params().items():
                if isinstance(param_value, (int, float, str, bool, type(None))):
                    mlflow.log_param(param_name, param_value)
            
            mlflow.log_param("training_samples", sample_size)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("training_time", train_time)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
    
    print("Benchmark complete!")

if __name__ == "__main__":
    main()