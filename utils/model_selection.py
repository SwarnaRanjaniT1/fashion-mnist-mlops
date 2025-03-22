import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import time

def select_model_with_automl(X_train, y_train, X_test, y_test, n_trials=5, time_budget=60):
    """
    Use AutoML to select the best model for the Fashion MNIST dataset
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        n_trials (int): Number of trials for AutoML
        time_budget (int): Time budget in seconds
        
    Returns:
        tuple: (best_model, all_models)
    """
    st.write("Running AutoML to select the best model...")
    
    # Define a dictionary to store all models and their scores
    all_models = {}
    
    # Create a progress bar
    progress_bar = st.progress(0)
    model_status = st.empty()
    
    # List of models to try
    models = [
        ('Dummy Classifier', DummyClassifier(strategy='most_frequent')),
        ('Logistic Regression', LogisticRegression(max_iter=500, random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # Try SVM if we don't have too many samples (it's slow on large datasets)
    if len(X_train) <= 10000:
        models.append(('SVM', SVC(probability=True, random_state=42)))
    
    # Try each model and evaluate
    for i, (name, model) in enumerate(models):
        model_status.write(f"Training and evaluating {name}...")
        progress_bar.progress((i) / (len(models) + 1))
        
        # Fit the model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict and evaluate
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_score = accuracy_score(y_test, model.predict(X_test))
        
        # Store the model and its scores
        all_models[model] = {
            'name': name,
            'train_score': train_score,
            'test_score': test_score,
            'train_time': train_time
        }
        
        # Display current results
        st.write(f"- {name}: Test Accuracy = {test_score:.4f}, Training Time = {train_time:.2f}s")
    
    # Manual AutoML alternative
    model_status.write("Running additional model evaluation...")
    progress_bar.progress(len(models) / (len(models) + 1))
    
    # Try an ensemble approach (stacking classifier)
    try:
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Find the best performing models to use in the stack
        sorted_models = sorted(all_models.items(), key=lambda x: x[1]['test_score'], reverse=True)
        top_models = [model for model, _ in sorted_models[:3] if not isinstance(model, DummyClassifier)]
        
        if len(top_models) >= 2:  # Need at least 2 models for stacking
            # Create a stacking ensemble
            estimators = [(f"model_{i}", model) for i, model in enumerate(top_models)]
            
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3
            )
            
            # Sample data if dataset is large
            sample_size = min(5000, len(X_train))
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_sample = X_train[indices]
            y_train_sample = y_train[indices]
            
            # Time and fit the stacking model
            start_time = time.time()
            stack.fit(X_train_sample, y_train_sample)
            train_time = time.time() - start_time
            
            # Evaluate the stacking model
            train_score = accuracy_score(y_train, stack.predict(X_train))
            test_score = accuracy_score(y_test, stack.predict(X_test))
            
            # Store the model and its scores
            all_models[stack] = {
                'name': 'Stacking Ensemble',
                'train_score': train_score,
                'test_score': test_score,
                'train_time': train_time
            }
            
            st.write(f"- Stacking Ensemble: Test Accuracy = {test_score:.4f}, Training Time = {train_time:.2f}s")
            
    except Exception as e:
        st.warning(f"Stacking ensemble encountered an error: {str(e)}")
    
    progress_bar.progress(1.0)
    model_status.write("Model selection completed!")
    
    # Find the best model
    best_model = max(all_models.items(), key=lambda x: x[1]['test_score'])[0]
    
    # Show summary
    st.subheader("Model Comparison Summary")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame([
        {
            'Model': model_info['name'],
            'Train Accuracy': model_info['train_score'],
            'Test Accuracy': model_info['test_score'],
            'Training Time (s)': model_info['train_time']
        }
        for model_info in all_models.values()
    ])
    
    # Display the results
    st.dataframe(results_df.sort_values('Test Accuracy', ascending=False))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train and test accuracy side by side
    bar_width = 0.35
    indices = np.arange(len(all_models))
    
    model_names = [info['name'] for info in all_models.values()]
    train_scores = [info['train_score'] for info in all_models.values()]
    test_scores = [info['test_score'] for info in all_models.values()]
    
    # Sort by test accuracy
    sort_idx = np.argsort(test_scores)
    model_names = [model_names[i] for i in sort_idx]
    train_scores = [train_scores[i] for i in sort_idx]
    test_scores = [test_scores[i] for i in sort_idx]
    
    # Plot bars
    bars1 = ax.bar(indices - bar_width/2, train_scores, bar_width, label='Train Accuracy')
    bars2 = ax.bar(indices + bar_width/2, test_scores, bar_width, label='Test Accuracy')
    
    # Add labels and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Return the best model and all model results
    return best_model, all_models
