import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.metrics import confusion_matrix
from .data_loader import get_class_names

def explain_model(model, X, y, n_samples=500):
    """
    Generate model explainability visualizations
    
    Args:
        model: Trained model
        X (np.ndarray): Input features
        y (np.ndarray): Target labels
        n_samples (int): Number of samples to use for explainability
    """
    # Use a subset of data for explainability to improve performance
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y
    
    # Get class names
    class_names = get_class_names()
    
    # 1. Feature Importance using permutation importance
    st.subheader("Feature Importance")
    
    with st.spinner("Calculating permutation feature importance..."):
        # Compute permutation importance on a small subset for speed
        perm_importance = permutation_importance(model, X_subset, y_subset, 
                                              n_repeats=5, random_state=42)
        
        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()[-20:]  # Top 20 features
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh([f"Feature {i}" for i in sorted_idx], 
               perm_importance.importances_mean[sorted_idx])
        ax.set_title("Permutation Feature Importance")
        ax.set_xlabel("Mean Importance")
        plt.tight_layout()
        st.pyplot(fig)
    
    # 2. Model Explainability with Partial Dependence Plots
    st.subheader("Feature Effect Analysis")
    
    with st.spinner("Calculating feature effects..."):
        try:
            # Get top features based on permutation importance
            top_features = sorted_idx[-5:] if len(sorted_idx) > 0 else range(min(5, X_subset.shape[1]))
            
            # If the model has native feature importances, display them
            if hasattr(model, 'feature_importances_'):
                st.write("**Native Feature Importances**")
                
                # Create a bar plot of feature importances
                fig, ax = plt.subplots(figsize=(12, 8))
                feature_importances = model.feature_importances_
                
                # Sort and select top features
                indices = np.argsort(feature_importances)[-20:]
                
                ax.barh([f"Feature {i}" for i in indices], 
                       feature_importances[indices])
                ax.set_title("Native Feature Importance")
                ax.set_xlabel("Importance")
                plt.tight_layout()
                st.pyplot(fig)
            
            # For tree models, we can visualize tree structure
            if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'tree_'):
                st.write("**Single Decision Tree Visualization**")
                
                from sklearn.tree import plot_tree
                
                # Use the first tree for visualization if it's a random forest or similar ensemble
                fig, ax = plt.subplots(figsize=(15, 10))
                plot_tree(model.estimators_[0], 
                          feature_names=[f"Feature {i}" for i in range(X_subset.shape[1])],
                          class_names=class_names,
                          filled=True, 
                          max_depth=3,  # Limit depth for better visualization
                          ax=ax)
                st.pyplot(fig)
            
            # Partial Dependence Plots for top features
            st.write("**Partial Dependence Plots**")
            
            # Use sklearn's partial_dependence
            # For efficiency, limit to a smaller subset of data
            sample_for_pdp = min(500, len(X_subset))
            X_pdp = X_subset[:sample_for_pdp]
            
            # For multi-class, show target classes separately
            if hasattr(model, 'classes_') and len(model.classes_) > 2:
                # Select a few classes to display
                classes_to_plot = model.classes_[:min(3, len(model.classes_))]
                
                for feature_idx in top_features:
                    st.write(f"**Partial Dependence for Feature {feature_idx}**")
                    
                    try:
                        # Compute partial dependence for each class
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        for i, class_idx in enumerate(classes_to_plot):
                            # For classification, specify target class
                            pdp_result = partial_dependence(
                                model, 
                                X_pdp, 
                                features=[feature_idx],
                                kind='average',
                                target=int(class_idx)
                            )
                            
                            # Plot partial dependence
                            ax.plot(pdp_result['values'][0], 
                                   pdp_result['average'][0], 
                                   label=f"Class {class_names[int(class_idx)]}")
                            
                        ax.set_xlabel(f"Feature {feature_idx} Value")
                        ax.set_ylabel("Partial Dependence")
                        ax.set_title(f"Partial Dependence Plot for Feature {feature_idx}")
                        ax.legend()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Could not compute partial dependence for feature {feature_idx}: {str(e)}")
            else:
                # For binary classification or regression
                for feature_idx in top_features:
                    st.write(f"**Partial Dependence for Feature {feature_idx}**")
                    
                    try:
                        # Compute partial dependence
                        pdp_result = partial_dependence(
                            model, 
                            X_pdp, 
                            features=[feature_idx],
                            kind='average'
                        )
                        
                        # Plot partial dependence
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(pdp_result['values'][0], pdp_result['average'][0])
                        ax.set_xlabel(f"Feature {feature_idx} Value")
                        ax.set_ylabel("Partial Dependence")
                        ax.set_title(f"Partial Dependence Plot for Feature {feature_idx}")
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Could not compute partial dependence for feature {feature_idx}: {str(e)}")
            
            # Show feature interactions (2D partial dependence) for a few top features
            if len(top_features) >= 2:
                st.write("**Feature Interaction Analysis**")
                
                # Select top 2 features for interaction plot
                feature1, feature2 = top_features[-1], top_features[-2]
                
                try:
                    # Compute 2D partial dependence
                    pdp_interact = partial_dependence(
                        model, 
                        X_pdp, 
                        features=[(feature1, feature2)],
                        kind='average'
                    )
                    
                    # Plot 2D partial dependence
                    fig, ax = plt.subplots(figsize=(10, 8))
                    X, Y = np.meshgrid(pdp_interact['values'][0][0], pdp_interact['values'][0][1])
                    Z = pdp_interact['average'][0].T
                    
                    contour = ax.contourf(X, Y, Z, cmap='viridis')
                    plt.colorbar(contour, ax=ax)
                    
                    ax.set_xlabel(f"Feature {feature1}")
                    ax.set_ylabel(f"Feature {feature2}")
                    ax.set_title(f"Interaction Between Feature {feature1} and Feature {feature2}")
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Could not compute feature interaction: {str(e)}")
                
        except Exception as e:
            st.error(f"Error generating feature effects: {str(e)}")
            st.info("Unable to compute feature effects. This might be due to model compatibility issues.")
    
    # 3. Decision Boundaries (for 2D visualization)
    st.subheader("Decision Boundary Visualization")
    
    with st.spinner("Generating decision boundary visualization..."):
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_subset)
        
        # Plot decision boundaries
        from matplotlib.colors import ListedColormap
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a meshgrid
        h = 0.1  # step size in the mesh
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create PCA transformer that works on single samples
        class PCATransformer:
            def __init__(self, pca, mean):
                self.pca = pca
                self.mean = mean
            
            def transform(self, X):
                X_restored = np.dot(X, self.pca.components_) + self.mean
                return X_restored
        
        # Predict on the meshgrid
        try:
            # First transform mesh points back to original feature space
            pca_transformer = PCATransformer(pca, X_subset.mean(axis=0))
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_points_original = pca_transformer.transform(mesh_points)
            
            # Make predictions
            Z = model.predict(mesh_points_original)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            cmap = plt.cm.tab10
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            
            # Plot training points
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_subset, 
                              s=40, edgecolors='k', cmap=cmap)
            
            # Add legend
            class_names = get_class_names()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        label=class_names[i], 
                                        markerfacecolor=cmap(i/10), 
                                        markersize=10) 
                             for i in range(10)]
            ax.legend(handles=legend_elements, title="Classes", loc="best")
            
            ax.set_title('Decision Boundary (PCA 2D projection)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating decision boundary visualization: {str(e)}")
            
            # If decision boundary fails, just show the PCA scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_subset, 
                              s=40, edgecolors='k', cmap=plt.cm.tab10)
            
            class_names = get_class_names()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        label=class_names[i], 
                                        markerfacecolor=plt.cm.tab10(i/10), 
                                        markersize=10) 
                             for i in range(10)]
            ax.legend(handles=legend_elements, title="Classes", loc="best")
            
            ax.set_title('Data Distribution (PCA 2D projection)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            
            st.pyplot(fig)
    
    # 4. Confusion Matrix Analysis
    st.subheader("Confusion Matrix Analysis")
    
    with st.spinner("Generating confusion matrix..."):
        
        # Predict on the subset
        y_pred = model.predict(X_subset)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_subset, y_pred)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax)
        
        # Add class labels
        class_names = get_class_names()
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.tight_layout()
        st.pyplot(fig)
    
    # 5. Misclassification Analysis
    st.subheader("Misclassification Analysis")
    
    with st.spinner("Analyzing misclassifications..."):
        # Find misclassified examples
        misclassified = y_pred != y_subset
        misclassified_indices = np.where(misclassified)[0]
        
        if len(misclassified_indices) > 0:
            # Create a DataFrame with misclassification details
            misclass_df = pd.DataFrame({
                'True Class': [class_names[y_subset[i]] for i in misclassified_indices],
                'Predicted Class': [class_names[y_pred[i]] for i in misclassified_indices],
            })
            
            # Compute misclassification statistics
            misclass_counts = misclass_df.groupby(['True Class', 'Predicted Class']).size().reset_index(name='Count')
            misclass_counts = misclass_counts.sort_values('Count', ascending=False)
            
            # Display the top misclassifications
            st.write("**Top Misclassifications**")
            st.dataframe(misclass_counts.head(10))
            
            # Visualize common misclassifications
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create a matrix of misclassification counts
            misclass_matrix = np.zeros((10, 10))
            for i, j, count in zip(misclass_counts['True Class'].map({name: i for i, name in enumerate(class_names)}),
                                 misclass_counts['Predicted Class'].map({name: i for i, name in enumerate(class_names)}),
                                 misclass_counts['Count']):
                misclass_matrix[i, j] = count
            
            im = ax.imshow(misclass_matrix, cmap='YlOrRd')
            plt.colorbar(im, ax=ax)
            
            # Add class labels
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
            
            ax.set_title('Misclassification Heat Map')
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            
            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    if misclass_matrix[i, j] > 0:
                        ax.text(j, i, int(misclass_matrix[i, j]),
                                ha="center", va="center",
                                color="white" if misclass_matrix[i, j] > misclass_matrix.max() / 2 else "black")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No misclassifications found in the analyzed subset!")
    
    # 6. Key Insights Summary
    st.subheader("Key Insights Summary")
    
    # Compile insights based on the visualizations
    insights = []
    
    # Top features
    if sorted_idx.size > 0:
        top_feature_indices = sorted_idx[-3:]  # Top 3 features
        insights.append(f"**Top Features**: Features {', '.join(map(str, top_feature_indices))} are the most important for classification.")
    
    # Misclassification patterns
    if 'misclass_counts' in locals() and len(misclass_counts) > 0:
        top_misclass = misclass_counts.iloc[0]
        insights.append(f"**Common Confusion**: The model often confuses {top_misclass['True Class']} with {top_misclass['Predicted Class']} ({top_misclass['Count']} instances).")
    
    # Model accuracy
    if 'y_pred' in locals() and 'y_subset' in locals():
        accuracy = (y_pred == y_subset).mean()
        insights.append(f"**Model Accuracy**: {accuracy:.2%} on the analyzed subset.")
    
    # Class imbalance insights
    if 'y_subset' in locals():
        class_counts = pd.Series(y_subset).value_counts()
        most_common = class_counts.idxmax()
        least_common = class_counts.idxmin()
        insights.append(f"**Class Distribution**: The most common class in the subset is {class_names[most_common]} ({class_counts[most_common]} instances) and the least common is {class_names[least_common]} ({class_counts[least_common]} instances).")
    
    # Display insights
    for insight in insights:
        st.markdown(insight)
    
    # Recommendations
    st.markdown("### Recommendations for Improving the Model")
    
    recommendations = [
        "Consider feature engineering techniques tailored to the top important features",
        "Explore data augmentation for classes with high misclassification rates",
        "Try ensemble methods to improve overall classification accuracy",
        "Implement class weighting if there's significant class imbalance",
        "Use feature importance and partial dependence plots to inform feature selection"
    ]
    
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(f"{i}. {recommendation}")
