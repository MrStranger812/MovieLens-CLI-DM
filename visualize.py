# Visualization script

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import gzip
from scipy import sparse
import warnings
import os
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting styles
plt.style.use('default')
sns.set_theme(style="whitegrid")

def load_processed_data():
    """Load processed data from files."""
    try:
        user_item_matrix = sparse.load_npz('data/processed/user_item_matrix.npz')
        print("Sparse matrix loaded successfully")
        
        with gzip.open('data/processed/ml_ready_datasets.pkl.gz', 'rb') as f:
            ml_datasets = pickle.load(f)
        print("ML datasets loaded successfully")
        
        with gzip.open('data/processed/hyper_features.pkl.gz', 'rb') as f:
            hyper_features = pickle.load(f)
        print("Hyper features loaded successfully")
        
        return user_item_matrix, ml_datasets, hyper_features
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def visualize_data_cleaning(ml_datasets):
    """Visualize data cleaning steps."""
    try:
        if ml_datasets is None:
            print("No ML datasets available for data cleaning visualization")
            return
            
        # Handle case where ml_datasets might be a dict or DataFrame
        data = ml_datasets if isinstance(ml_datasets, pd.DataFrame) else (
            ml_datasets.get('train') if isinstance(ml_datasets, dict) and 'train' in ml_datasets else None
        )
        if data is None or not isinstance(data, pd.DataFrame):
            print("Data for cleaning visualization must be a pandas DataFrame")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Cleaning Analysis', fontsize=16)
        
        # Plot 1: Missing Values Distribution
        missing = data.isnull().mean()
        axes[0, 0].bar(missing.index, missing.values)
        axes[0, 0].set_title('Missing Values Distribution')
        axes[0, 0].set_xticks(range(len(missing)))
        axes[0, 0].set_xticklabels(missing.index, rotation=90)
        axes[0, 0].set_ylabel('Fraction Missing')
        
        # Plot 2: Outlier Detection (boxplot of first numerical column)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            sns.boxplot(y=data[numerical_cols[0]], ax=axes[0, 1])
            axes[0, 1].set_title(f'Outliers: {numerical_cols[0]}')
        
        # Plot 3: Feature Correlation Matrix (limit to first 5 numerical cols for readability)
        if len(numerical_cols) > 1:
            corr = data[numerical_cols[:5]].corr()
            sns.heatmap(corr, ax=axes[1, 0], annot=True, cmap='coolwarm')
            axes[1, 0].set_title('Feature Correlation Matrix')
        
        # Plot 4: Data Type Distribution
        dtypes = data.dtypes.value_counts()
        axes[1, 1].bar(dtypes.index.astype(str), dtypes.values)
        axes[1, 1].set_title('Data Type Distribution')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('data_cleaning_analysis.png')
        plt.close()
        print("Data cleaning visualization saved as 'data_cleaning_analysis.png'")
    except Exception as e:
        print(f"Error in data cleaning visualization: {str(e)}")

def visualize_feature_engineering(ml_datasets):
    """Visualize feature engineering results."""
    try:
        if ml_datasets is None:
            print("No ML datasets available for feature engineering visualization")
            return
            
        # Handle case where ml_datasets might be a dict or DataFrame
        data = ml_datasets if isinstance(ml_datasets, pd.DataFrame) else (
            ml_datasets.get('train') if isinstance(ml_datasets, dict) and 'train' in ml_datasets else None
        )
        if data is None or not isinstance(data, pd.DataFrame):
            print("Data for feature engineering visualization must be a pandas DataFrame")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Engineering Analysis', fontsize=16)
        
        # Plot 1: Feature Distribution (histogram of first numerical feature)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            sns.histplot(data[numerical_cols[0]], ax=axes[0, 0], bins=30)
            axes[0, 0].set_title(f'Distribution: {numerical_cols[0]}')
        
        # Plot 2: Feature Correlations (limit to first 5 cols)
        if len(numerical_cols) > 1:
            corr = data[numerical_cols[:5]].corr()
            sns.heatmap(corr, ax=axes[0, 1], annot=True, cmap='coolwarm')
            axes[0, 1].set_title('Feature Correlations')
        
        # Plot 3: Feature Means (numerical features)
        if len(numerical_cols) > 0:
            means = data[numerical_cols].mean()
            axes[1, 0].bar(means.index, means.values)
            axes[1, 0].set_title('Mean of Numerical Features')
            axes[1, 0].set_xticks(range(len(means)))
            axes[1, 0].set_xticklabels(means.index, rotation=90)
        
        # Plot 4: Feature Std Dev (numerical features)
        if len(numerical_cols) > 0:
            stds = data[numerical_cols].std()
            axes[1, 1].bar(stds.index, stds.values)
            axes[1, 1].set_title('Std Dev of Numerical Features')
            axes[1, 1].set_xticks(range(len(stds)))
            axes[1, 1].set_xticklabels(stds.index, rotation=90)
        
        plt.tight_layout()
        plt.savefig('feature_engineering_analysis.png')
        plt.close()
        print("Feature engineering visualization saved as 'feature_engineering_analysis.png'")
    except Exception as e:
        print(f"Error in feature engineering visualization: {str(e)}")

def visualize_pca_results(ml_datasets):
    """Visualize PCA results."""
    try:
        if ml_datasets is None:
            print("No ML datasets available for PCA visualization")
            return
            
        # Handle case where ml_datasets might be a dict or DataFrame
        data = ml_datasets if isinstance(ml_datasets, pd.DataFrame) else (
            ml_datasets.get('train') if isinstance(ml_datasets, dict) and 'train' in ml_datasets else None
        )
        if data is None or not isinstance(data, pd.DataFrame):
            print("Data for PCA visualization must be a pandas DataFrame")
            return

        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            print("Not enough numerical features for PCA")
            return
            
        # Prepare data for PCA (drop NaNs)
        data_numerical = data[numerical_cols].dropna()
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_numerical)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PCA Analysis', fontsize=16)
        
        # Plot 1: Explained Variance Ratio
        axes[0].bar(range(1, 3), pca.explained_variance_ratio_)
        axes[0].set_title('Explained Variance Ratio')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Ratio')
        axes[0].set_xticks([1, 2])
        
        # Plot 2: PCA Scatter Plot
        axes[1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        axes[1].set_title('PCA Scatter Plot')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png')
        plt.close()
        print("PCA visualization saved as 'pca_analysis.png'")
    except Exception as e:
        print(f"Error in PCA visualization: {str(e)}")

def visualize_user_item_matrix(user_item_matrix):
    """Visualize user-item matrix characteristics."""
    try:
        if user_item_matrix is None:
            print("No user-item matrix available for visualization")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('User-Item Matrix Analysis', fontsize=16)
        
        # Plot 1: Matrix Sparsity
        sparsity = 1.0 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
        axes[0].bar(['Sparsity'], [sparsity])
        axes[0].set_title('Matrix Sparsity')
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel('Fraction')
        
        # Plot 2: Rating Distribution
        ratings = user_item_matrix.data
        sns.histplot(ratings, ax=axes[1], bins=30)
        axes[1].set_title('Rating Distribution')
        axes[1].set_xlabel('Rating')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('user_item_matrix_analysis.png')
        plt.close()
        print("User-item matrix visualization saved as 'user_item_matrix_analysis.png'")
    except Exception as e:
        print(f"Error in user-item matrix visualization: {str(e)}")

def main():
    """Main function to run all visualizations."""
    print("Loading processed data...")
    user_item_matrix, ml_datasets, hyper_features = load_processed_data()
    
    # Debug prints to inspect data
    if user_item_matrix is not None:
        print("User-item matrix sample (first 5x5):\n", user_item_matrix[:5, :5].toarray())
        ratings = user_item_matrix.data
        hist, bins = np.histogram(ratings, bins=np.arange(0.5, 5.5, 0.5))
        print("Rating histogram:", hist)
    
    print("Type of ml_datasets:", type(ml_datasets))
    if isinstance(ml_datasets, dict):
        print("Keys in ml_datasets:", list(ml_datasets.keys()))
    elif isinstance(ml_datasets, pd.DataFrame):
        print("Shape of ml_datasets:", ml_datasets.shape)
        print("Columns:", ml_datasets.columns.tolist())
    
    print("Type of hyper_features:", type(hyper_features))
    
    # Check if data loaded successfully
    if user_item_matrix is None or ml_datasets is None or hyper_features is None:
        print("Failed to load data. Please check the data files exist and are properly formatted.")
        return
        
    print("Creating visualizations...")
    visualize_data_cleaning(ml_datasets)
    visualize_feature_engineering(ml_datasets)
    visualize_pca_results(ml_datasets)
    visualize_user_item_matrix(user_item_matrix)
    print("Visualizations completed. Check the generated PNG files in the current directory.")

if __name__ == "__main__":
    main()