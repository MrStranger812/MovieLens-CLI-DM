import pickle
import gzip
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def add_pca_to_existing_features():
    """Add PCA to existing feature files."""
    
    PROCESSED_DATA_DIR = Path("data/processed")
    
    # Load existing features
    print("Loading existing features...")
    with gzip.open(PROCESSED_DATA_DIR / "hyper_features.pkl.gz", 'rb') as f:
        features = pickle.load(f)
    
    user_features = features.get('user_features')
    movie_features = features.get('movie_features')
    
    if user_features is None or movie_features is None:
        print("Error: Could not find user or movie features")
        return
    
    # Apply PCA to user features
    print(f"Applying PCA to user features ({user_features.shape})...")
    user_numeric = user_features.select_dtypes(include=[np.number]).fillna(0)
    user_scaler = StandardScaler()
    user_scaled = user_scaler.fit_transform(user_numeric)
    
    user_pca = PCA(n_components=min(50, user_scaled.shape[1]), random_state=42)
    user_pca_features = user_pca.fit_transform(user_scaled)
    
    user_pca_df = pd.DataFrame(
        user_pca_features,
        index=user_features.index,
        columns=[f'user_pc_{i+1}' for i in range(user_pca_features.shape[1])]
    )
    
    print(f"User PCA: {user_pca.explained_variance_ratio_.sum():.3f} variance explained")
    
    # Apply PCA to movie features
    print(f"Applying PCA to movie features ({movie_features.shape})...")
    movie_numeric = movie_features.select_dtypes(include=[np.number]).fillna(0)
    movie_scaler = StandardScaler()
    movie_scaled = movie_scaler.fit_transform(movie_numeric)
    
    movie_pca = PCA(n_components=min(50, movie_scaled.shape[1]), random_state=42)
    movie_pca_features = movie_pca.fit_transform(movie_scaled)
    
    movie_pca_df = pd.DataFrame(
        movie_pca_features,
        index=movie_features.index,
        columns=[f'movie_pc_{i+1}' for i in range(movie_pca_features.shape[1])]
    )
    
    print(f"Movie PCA: {movie_pca.explained_variance_ratio_.sum():.3f} variance explained")
    
    # Save PCA features
    user_pca_df.to_parquet(PROCESSED_DATA_DIR / "user_pca_features.parquet", compression='snappy')
    movie_pca_df.to_parquet(PROCESSED_DATA_DIR / "movie_pca_features.parquet", compression='snappy')
    
    # Save PCA models
    pca_models = {
        'user_pca': user_pca,
        'movie_pca': movie_pca,
        'user_scaler': user_scaler,
        'movie_scaler': movie_scaler
    }
    
    with open(PROCESSED_DATA_DIR / "pca_models.pkl", 'wb') as f:
        pickle.dump(pca_models, f)
    
    print("✓ PCA features saved successfully!")

if __name__ == "__main__":
    add_pca_to_existing_features()