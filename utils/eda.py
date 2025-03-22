import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import io
import base64
from .data_loader import get_class_names

def perform_eda(X_train, y_train, sample_size=5000):
    """
    Perform Exploratory Data Analysis on the Fashion MNIST dataset
    
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        sample_size (int): Number of samples to use for the analysis
    
    Returns:
        dict: Dictionary containing EDA results
    """
    # Use a sample of the data for faster processing
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]
    else:
        X_sample = X_train
        y_sample = y_train
    
    # Flatten images for analysis
    X_flat = X_sample.reshape(X_sample.shape[0], -1)
    
    # Create a pandas DataFrame for analysis
    df = pd.DataFrame(X_flat)
    df['label'] = y_sample
    
    # Basic statistics
    basic_stats = {
        'num_samples': len(X_train),
        'image_shape': X_train[0].shape,
        'class_distribution': {i: int((y_train == i).sum()) for i in range(10)},
        'class_names': get_class_names(),
        'pixel_min': float(X_train.min()),
        'pixel_max': float(X_train.max()),
        'pixel_mean': float(X_train.mean()),
        'pixel_std': float(X_train.std())
    }
    
    # Compute class distribution
    class_dist = pd.Series(y_train).value_counts().sort_index()
    
    # Generate class-specific statistics
    class_stats = {}
    for i in range(10):
        class_images = X_train[y_train == i]
        class_stats[i] = {
            'count': int(len(class_images)),
            'mean_pixel_value': float(class_images.mean()),
            'std_pixel_value': float(class_images.std())
        }
    
    # Perform dimensionality reduction for visualization
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    
    # t-SNE (on a smaller subset for speed)
    if len(X_flat) > 1000:
        X_subset = X_flat[:1000]
        y_subset = y_sample[:1000]
    else:
        X_subset = X_flat
        y_subset = y_sample
        
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)
    
    # Calculate correlation between pixels
    # Only use a small subset of pixels to avoid memory issues
    pixels_sample = np.random.choice(X_flat.shape[1], min(100, X_flat.shape[1]), replace=False)
    pixel_corr = np.corrcoef(X_flat[:, pixels_sample].T)
    
    # Generate additional feature statistics
    feature_stats = {}
    for i in range(min(20, X_flat.shape[1])):  # Analyze up to 20 features
        feature_stats[i] = {
            'mean': float(X_flat[:, i].mean()),
            'std': float(X_flat[:, i].std()),
            'min': float(X_flat[:, i].min()),
            'max': float(X_flat[:, i].max()),
            'median': float(np.median(X_flat[:, i]))
        }
    
    # Compile all EDA results
    eda_results = {
        'basic_stats': basic_stats,
        'class_distribution': class_dist,
        'class_stats': class_stats,
        'pca_results': {'X_pca': X_pca, 'y': y_sample[:len(X_pca)]},
        'tsne_results': {'X_tsne': X_tsne, 'y': y_subset[:len(X_tsne)]},
        'pixel_correlation': pixel_corr,
        'feature_stats': feature_stats
    }
    
    return eda_results

def display_eda_summary(eda_results):
    """
    Display the EDA results in Streamlit
    
    Args:
        eda_results (dict): Dictionary containing EDA results from perform_eda()
    """
    # Display basic statistics
    st.subheader("Dataset Overview")
    stats = eda_results['basic_stats']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Samples", f"{stats['num_samples']:,}")
    with col2:
        st.metric("Image Shape", f"{stats['image_shape'][0]}x{stats['image_shape'][1]}")
    with col3:
        st.metric("Number of Classes", "10")
    
    # Display pixel value statistics
    st.subheader("Pixel Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Pixel Value", f"{stats['pixel_min']:.3f}")
    with col2:
        st.metric("Max Pixel Value", f"{stats['pixel_max']:.3f}")
    with col3:
        st.metric("Mean Pixel Value", f"{stats['pixel_mean']:.3f}")
    with col4:
        st.metric("Std Pixel Value", f"{stats['pixel_std']:.3f}")
    
    # Display class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = stats['class_names']
    class_dist = eda_results['class_distribution']
    
    bars = ax.bar(range(10), class_dist, color='skyblue')
    ax.set_xticks(range(10))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    
    # Add labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display dimensionality reduction visualizations
    st.subheader("Dimensionality Reduction Visualizations")
    
    col1, col2 = st.columns(2)
    
    # PCA plot
    with col1:
        st.write("**PCA Visualization**")
        X_pca = eda_results['pca_results']['X_pca']
        y_pca = eda_results['pca_results']['y']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pca, cmap='tab10', alpha=0.7, s=30)
        ax.set_title('PCA Visualization')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        # Create a custom legend
        class_names = stats['class_names']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      label=class_names[i], 
                                      markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                      markersize=10) 
                           for i in range(10)]
        ax.legend(handles=legend_elements, title="Classes", loc="best", ncol=2)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # t-SNE plot
    with col2:
        st.write("**t-SNE Visualization**")
        X_tsne = eda_results['tsne_results']['X_tsne']
        y_tsne = eda_results['tsne_results']['y']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='tab10', alpha=0.7, s=30)
        ax.set_title('t-SNE Visualization')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        
        # Create a custom legend
        class_names = stats['class_names']
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      label=class_names[i], 
                                      markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                      markersize=10) 
                           for i in range(10)]
        ax.legend(handles=legend_elements, title="Classes", loc="best", ncol=2)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display class-specific statistics
    st.subheader("Class-Specific Statistics")
    
    class_stats = eda_results['class_stats']
    class_names = stats['class_names']
    
    # Create a DataFrame for better display
    class_stats_df = pd.DataFrame({
        'Class': class_names,
        'Count': [class_stats[i]['count'] for i in range(10)],
        'Mean Pixel Value': [class_stats[i]['mean_pixel_value'] for i in range(10)],
        'Std Pixel Value': [class_stats[i]['std_pixel_value'] for i in range(10)]
    })
    
    st.dataframe(class_stats_df)
    
    # Display pixel correlation heatmap
    st.subheader("Pixel Correlation Analysis")
    
    pixel_corr = eda_results['pixel_correlation']
    
    # Show a subset of the correlation matrix for better visualization
    subset_size = min(30, pixel_corr.shape[0])
    pixel_corr_subset = pixel_corr[:subset_size, :subset_size]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pixel_corr_subset, cmap='coolwarm')
    ax.set_title(f'Pixel Correlation Matrix (Subset of {subset_size}x{subset_size})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature statistics
    st.subheader("Feature-Specific Statistics")
    
    feature_stats = eda_results['feature_stats']
    
    # Create a DataFrame for better display
    feature_stats_df = pd.DataFrame({
        'Feature': [f'Feature {i}' for i in feature_stats.keys()],
        'Mean': [feature_stats[i]['mean'] for i in feature_stats.keys()],
        'Std': [feature_stats[i]['std'] for i in feature_stats.keys()],
        'Min': [feature_stats[i]['min'] for i in feature_stats.keys()],
        'Max': [feature_stats[i]['max'] for i in feature_stats.keys()],
        'Median': [feature_stats[i]['median'] for i in feature_stats.keys()]
    })
    
    st.dataframe(feature_stats_df)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Plot histograms for first few features
    num_features_to_plot = min(6, len(feature_stats))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature_idx in enumerate(list(feature_stats.keys())[:num_features_to_plot]):
        sns.histplot(x=eda_results['pca_results']['X_pca'][:, 0 if i==0 else 1 if i==1 else 0], 
                    hue=eda_results['pca_results']['y'], 
                    bins=30, 
                    ax=axes[i], 
                    kde=True,
                    palette='tab10')
        axes[i].set_title(f'{"PCA Component 1" if i==0 else "PCA Component 2" if i==1 else f"Feature {feature_idx}"} Distribution')
        axes[i].set_xlabel(f'{"PCA Component 1" if i==0 else "PCA Component 2" if i==1 else f"Feature {feature_idx}"}')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    st.pyplot(fig)
