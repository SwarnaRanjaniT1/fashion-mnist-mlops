import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def optimize_hyperparameters(model, X_train, y_train, X_test, y_test, n_trials=30):
    """
    Optimize hyperparameters for the given model using GridSearchCV
    
    Args:
        model: The model to optimize
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        n_trials (int): Not used in this implementation, kept for API compatibility
        
    Returns:
        tuple: (optimized_model, grid_search_results)
    """
    st.write(f"Optimizing hyperparameters for {type(model).__name__}...")
    
    # Define parameter grids based on model type
    param_grid = {}
    
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': [42]
        }
        
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'random_state': [42]
        }
        
    elif isinstance(model, SVC):
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'probability': [True],
            'random_state': [42]
        }
        
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
    elif isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000],
            'random_state': [42]
        }
    
    # If the model type is not recognized, use a minimal param grid
    if not param_grid:
        st.warning(f"Model type {type(model).__name__} not recognized for hyperparameter optimization.")
        # Create a dummy grid search that just returns the original model
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid={'random_state': [42]},
            cv=3,
            scoring='accuracy',
            return_train_score=True
        )
        grid_search.fit(X_train[:100], y_train[:100])  # Use small sample to be fast
        return model, grid_search
    
    # For faster optimization, use a subset of data if it's large
    if len(X_train) > 10000:
        # Use stratified sampling
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, 
            train_size=5000,  # Use smaller sample for faster grid search
            stratify=y_train, 
            random_state=42
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    # Create a progress placeholder
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Set up GridSearchCV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Create grid search with the appropriate parameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        verbose=1
    )
    
    # Track progress manually since GridSearchCV doesn't have native progress tracking
    status_placeholder.write("Running grid search...")
    total_combinations = np.prod([len(values) for values in param_grid.values()])
    progress_placeholder.progress(0)
    
    # Store results for plotting
    all_results = []
    start_time = time.time()

    # Run the grid search
    try:
        grid_search.fit(X_sample, y_sample)
        
        # Get all results for visualization
        results = pd.DataFrame(grid_search.cv_results_)
        all_results = results.sort_values('mean_test_score', ascending=False)
    except Exception as e:
        st.error(f"Error in grid search: {str(e)}")
        # Return original model if grid search fails
        return model, None
    
    # Cleanup progress indicators
    progress_placeholder.progress(1.0)
    time.sleep(0.5)  # Short delay to show completed progress
    progress_placeholder.empty()
    status_placeholder.empty()
    
    # Get the best parameters
    best_params = grid_search.best_params_
    st.write("Best hyperparameters found:")
    for param, value in best_params.items():
        st.write(f"- **{param}**: {value}")
    
    # Create and train the optimized model on the full dataset
    optimized_model = type(model)(**best_params)
    
    with st.spinner("Training the optimized model on the full dataset..."):
        optimized_model.fit(X_train, y_train)
    
    # Evaluate the optimized model
    y_pred = optimized_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Optimized model test accuracy: {accuracy:.4f}")
    
    # Compare with the original model
    original_model = model.fit(X_train, y_train)
    original_accuracy = accuracy_score(y_test, original_model.predict(X_test))
    
    st.write(f"Original model test accuracy: {original_accuracy:.4f}")
    
    # Calculate improvement
    improvement = (accuracy - original_accuracy) / original_accuracy * 100
    
    if improvement > 0:
        st.success(f"Hyperparameter optimization improved accuracy by {improvement:.2f}%!")
    else:
        st.info("Hyperparameter optimization didn't improve the model. The original model was already well-tuned.")
    
    # Visualization of search results
    if all_results is not None and len(all_results) > 0:
        st.subheader("Grid Search Results")
        
        # Display top results table
        st.write("Top 5 parameter combinations:")
        top_results = all_results[['params', 'mean_test_score', 'std_test_score']].head(5)
        st.dataframe(top_results)
        
        # Plot the distribution of scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_results['mean_test_score'], bins=20, alpha=0.7)
        ax.axvline(grid_search.best_score_, color='red', linestyle='--', 
                label=f'Best score: {grid_search.best_score_:.4f}')
        
        ax.set_xlabel('Mean Test Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Test Scores in Grid Search')
        ax.legend()
        st.pyplot(fig)
        
        # If we have enough data points, create parameter importance visual
        if len(param_grid) > 1 and len(all_results) > 5:
            st.subheader("Parameter Importance Analysis")
            
            # Basic parameter importance by grouping results
            fig, axes = plt.subplots(nrows=min(3, len(param_grid)), ncols=1, 
                                    figsize=(10, 3*min(3, len(param_grid))))
            
            # If we have only one parameter, ensure axes is in a list
            if len(param_grid) == 1:
                axes = [axes]
            
            # Plot performance across different parameter values
            for i, (param_name, param_values) in enumerate(list(param_grid.items())[:3]):  # Show only up to 3 params
                if i >= len(axes):
                    break
                    
                if len(param_values) > 1:  # Only analyze if parameter has multiple values
                    # Group by parameter value and compute mean score
                    param_importance = all_results.groupby(f'param_{param_name}')['mean_test_score'].mean()
                    
                    # Sort by parameter value for better visualization if numeric
                    try:
                        param_importance = param_importance.sort_index()
                    except:
                        pass
                    
                    # Plot
                    param_importance.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'Mean Score by {param_name}')
                    axes[i].set_ylabel('Mean Test Score')
                    axes[i].set_xlabel(param_name)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Create a class to hold the grid search information for compatibility with the rest of the code
    class GridSearchResults:
        def __init__(self, grid_search):
            self.grid_search = grid_search
            self.best_trial = type('obj', (object,), {'params': grid_search.best_params_})
            self.trials = []
            
            # Add each grid search result as a trial
            results = pd.DataFrame(grid_search.cv_results_)
            for i, row in results.iterrows():
                trial = type('obj', (object,), {
                    'number': i,
                    'value': row['mean_test_score'],
                    'params': row['params']
                })
                self.trials.append(trial)
    
    # Return the optimized model and grid search results wrapped to match expected interface
    return optimized_model, GridSearchResults(grid_search)
