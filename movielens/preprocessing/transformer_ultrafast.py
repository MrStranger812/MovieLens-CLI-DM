import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix
from scipy import sparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from joblib import Parallel, delayed
import multiprocessing as mp
import warnings
import gc
import numba
from numba import jit, prange, njit
import pickle
from sklearn.utils import murmurhash3_32

# GPU imports with fallbacks
try:
    import cupy as cp
    import cuml
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    from cuml.decomposition import TruncatedSVD as CuTruncatedSVD
    from cuml.cluster import KMeans as CuKMeans
    import rmm
    GPU_AVAILABLE = True
    print("[bold green]✓ GPU acceleration enabled with CuPy and cuML[/bold green]")
except ImportError as e:
    cp = np
    GPU_AVAILABLE = False
    print(f"[yellow]⚠ GPU libraries not available: {e}[/yellow]")
    print("[yellow]Falling back to CPU-only processing[/yellow]")

warnings.filterwarnings('ignore')

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup for 4GB VRAM."""
    
    def __init__(self, vram_limit_gb=4.0):
        self.gpu_available = GPU_AVAILABLE
        self.vram_limit_gb = vram_limit_gb
        self.safety_factor = 0.6  # Use only 60% of VRAM for safety
        
        if self.gpu_available:
            try:
                # Set memory limit
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=int(vram_limit_gb * 1024**3 * self.safety_factor))
                
                self.console = Console()
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                self.console.print(f"[green]✓ GPU Memory initialized: {total_mem / 1024**3:.1f}GB total, using {vram_limit_gb * self.safety_factor:.1f}GB limit[/green]")
            except Exception as e:
                self.gpu_available = False
                print(f"[yellow]GPU memory setup failed: {e}[/yellow]")
    
    def get_available_memory(self):
        """Get available GPU memory in bytes."""
        if not self.gpu_available:
            return 0
        try:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            return free_mem
        except:
            return 0
    
    def clear_cache(self):
        """Aggressively clear GPU memory cache."""
        if self.gpu_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                if hasattr(cp, '_default_memory_pool'):
                    cp._default_memory_pool.free_all_blocks()
                gc.collect()
                cp.cuda.MemoryPool().free_all_blocks()
            except:
                pass
    
    def can_fit_array(self, shape, dtype=cp.float32):
        """Check if array of given shape can fit in GPU memory."""
        if not self.gpu_available:
            return False
        required_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        available_bytes = self.get_available_memory()
        # Be conservative with 4GB VRAM
        return required_bytes < (available_bytes * 0.5)

# CPU-optimized numba functions
@njit(parallel=True, cache=True, fastmath=True)
def ultra_fast_groupby_stats(user_ids, ratings, movie_ids, timestamps, n_users):
    """Ultra-fast groupby with all statistics in one pass."""
    # Pre-allocate arrays
    counts = np.zeros(n_users, dtype=np.int32)
    sums = np.zeros(n_users, dtype=np.float32)
    sum_squares = np.zeros(n_users, dtype=np.float32)
    mins = np.full(n_users, 5.0, dtype=np.float32)
    maxs = np.zeros(n_users, dtype=np.float32)
    first_time = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
    last_time = np.zeros(n_users, dtype=np.int64)
    unique_counts = np.zeros(n_users, dtype=np.int32)
    
    # Single pass through data
    for i in prange(len(user_ids)):
        uid = user_ids[i]
        rating = ratings[i]
        timestamp = timestamps[i]
        
        counts[uid] += 1
        sums[uid] += rating
        sum_squares[uid] += rating * rating
        
        if rating < mins[uid]:
            mins[uid] = rating
        if rating > maxs[uid]:
            maxs[uid] = rating
            
        if timestamp < first_time[uid]:
            first_time[uid] = timestamp
        if timestamp > last_time[uid]:
            last_time[uid] = timestamp
            
        unique_counts[uid] += 1  # Simplified for speed
    
    # Calculate derived statistics
    means = np.zeros(n_users, dtype=np.float32)
    stds = np.zeros(n_users, dtype=np.float32)
    
    for i in prange(n_users):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
            if counts[i] > 1:
                variance = (sum_squares[i] / counts[i]) - (means[i] * means[i])
                if variance > 0:
                    stds[i] = np.sqrt(variance)
    
    return counts, means, stds, mins, maxs, first_time, last_time, unique_counts

class HyperOptimizedDataTransformer:
    """Optimized transformer for 4GB VRAM GPUs."""
    
    def __init__(self, n_jobs: int = None, use_gpu: bool = True):
        self.console = Console()
        self.n_jobs = n_jobs if n_jobs else mp.cpu_count()
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize GPU memory manager with 4GB limit
        self.gpu_manager = GPUMemoryManager(vram_limit_gb=4.0)
        self.use_gpu = self.use_gpu and self.gpu_manager.gpu_available
        
        # Batch sizes optimized for 4GB VRAM
        self.gpu_batch_size = 50000  # Smaller batches for 4GB
        self.feature_batch_size = 10000
        
        self.scalers = {}
        self.encoders = {}
        
        # Pre-compile numba functions
        self._precompile_numba()
        
        pd.options.mode.chained_assignment = None
        
        gpu_status = "GPU-ENABLED (4GB)" if self.use_gpu else "CPU-ONLY"
        self.console.print(f"[green]✓ {gpu_status} Transformer initialized with {self.n_jobs} cores[/green]")
    
    def _precompile_numba(self):
        """Pre-compile numba functions."""
        dummy_data = np.array([0], dtype=np.int32)
        dummy_ratings = np.array([3.5], dtype=np.float32)
        try:
            ultra_fast_groupby_stats(dummy_data, dummy_ratings, dummy_data, dummy_data, 1)
        except:
            pass
    
    def create_user_features_optimized(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create user features with CPU optimization (GPU causes issues with 4GB)."""
        self.console.print("[cyan]Creating user features (CPU-optimized for stability)...[/cyan]")
        
        # Convert to numpy arrays for speed
        unique_users = ratings_df['userId'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        
        # Vectorized conversion
        user_indices = np.array([user_to_idx[uid] for uid in ratings_df['userId'].values], dtype=np.int32)
        ratings = ratings_df['rating'].values.astype(np.float32)
        movie_ids = ratings_df['movieId'].values.astype(np.int32)
        timestamps = ratings_df['timestamp'].astype(np.int64).values // 10**9
        
        # Use CPU-optimized function
        counts, means, stds, mins, maxs, first_time, last_time, unique_counts = \
            ultra_fast_groupby_stats(user_indices, ratings, movie_ids, timestamps, len(unique_users))
        
        # Create DataFrame
        user_features = pd.DataFrame({
            'rating_count': counts,
            'rating_mean': means,
            'rating_std': stds,
            'rating_min': mins,
            'rating_max': maxs,
            'movie_diversity': unique_counts,
            'first_timestamp': first_time,
            'last_timestamp': last_time
        }, index=unique_users)
        
        # Derived features
        user_features['time_span_days'] = (user_features['last_timestamp'] - user_features['first_timestamp']) / 86400
        user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
        user_features['rating_frequency'] = user_features['rating_count'] / (user_features['time_span_days'] + 1)
        
        # Binary features
        q75_count = np.percentile(counts, 75)
        user_features['is_active'] = (user_features['rating_count'] > q75_count).astype(np.int8)
        
        # Drop temporary columns
        user_features = user_features.drop(['first_timestamp', 'last_timestamp'], axis=1)
        
        # Optimize dtypes
        for col in user_features.select_dtypes(include=['float64']).columns:
            user_features[col] = user_features[col].astype(np.float32)
        
        gc.collect()
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features[/green]")
        return user_features
    
    def create_movie_features_optimized(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Create movie features with memory optimization."""
        self.console.print("[cyan]Creating movie features (memory-optimized)...[/cyan]")
        
        # Aggregate ratings per movie
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).round(4)
        
        movie_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'user_diversity']
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        
        # Merge with movie metadata
        movie_features = movie_stats.join(movies_df.set_index('movieId'), how='left')
        
        # Simple genre encoding (memory efficient)
        if 'genre_list' in movie_features.columns:
            # Count genres instead of full encoding
            movie_features['genre_count'] = movie_features['genre_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Top genres only
            all_genres = []
            for genres in movie_features['genre_list'].dropna():
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            top_genres = pd.Series(all_genres).value_counts().head(10).index
            
            for genre in top_genres:
                movie_features[f'genre_{genre.lower()}'] = movie_features['genre_list'].apply(
                    lambda x: 1 if isinstance(x, list) and genre in x else 0
                ).astype(np.int8)
        
        # Derived features
        movie_features['popularity_score'] = movie_features['rating_count'] * movie_features['rating_mean']
        movie_features['is_popular'] = (movie_features['rating_count'] > movie_features['rating_count'].quantile(0.8)).astype(np.int8)
        movie_features['is_acclaimed'] = (movie_features['rating_mean'] > 4.0).astype(np.int8)
        
        # Movie age
        current_year = 2025
        movie_features['movie_age'] = current_year - movie_features['year'].fillna(current_year)
        
        # Optimize dtypes
        movie_features = movie_features.fillna(0)
        for col in movie_features.select_dtypes(include=['float64']).columns:
            movie_features[col] = movie_features[col].astype(np.float32)
        
        gc.collect()
        self.console.print(f"[green]✓ Created {len(movie_features.columns)} movie features[/green]")
        return movie_features
    
    def create_ml_datasets(self, ratings_df: pd.DataFrame, 
                            user_features: pd.DataFrame,
                            movie_features: pd.DataFrame) -> Dict:
            """Create ML datasets focusing on regression and classification only."""
            self.console.print("[cyan]Creating ML datasets (regression & classification only)...[/cyan]")
            
            ml_datasets = {}
            
            try:
                # Validate and clean features first
                self.console.print("[cyan]Validating feature data types...[/cyan]")
                
                # Clean user features - only numeric columns
                user_clean = user_features.select_dtypes(include=[np.number]).copy()
                user_clean = user_clean.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Remove problematic columns
                problematic_cols = ['timestamp_min', 'timestamp_max']
                user_clean = user_clean.drop(columns=[col for col in problematic_cols if col in user_clean.columns])
                
                # Clean movie features - only numeric columns
                movie_clean = movie_features.select_dtypes(include=[np.number]).copy()
                movie_clean = movie_clean.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Remove problematic columns
                movie_clean = movie_clean.drop(columns=[col for col in problematic_cols if col in movie_clean.columns])
                
                self.console.print(f"[green]User features: {user_clean.shape[1]} numeric columns[/green]")
                self.console.print(f"[green]Movie features: {movie_clean.shape[1]} numeric columns[/green]")
                
                # Sample data for memory efficiency with 4GB VRAM
                sample_size = min(800_000, len(ratings_df))  # Reduced for 4GB
                if len(ratings_df) > sample_size:
                    self.console.print(f"[yellow]Sampling {sample_size:,} ratings for ML datasets (4GB VRAM limit)[/yellow]")
                    ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
                else:
                    ratings_sample = ratings_df
                
                # Build feature matrix step by step to control memory
                self.console.print("[cyan]Building feature matrix...[/cyan]")
                
                # Start with core data
                feature_matrix = ratings_sample[['userId', 'movieId', 'rating']].copy()
                
                # Limit features to prevent memory issues (4GB VRAM constraint)
                max_user_features = 30
                max_movie_features = 30
                
                # Add user features via efficient mapping
                user_cols = user_clean.columns[:max_user_features].tolist()
                self.console.print(f"[cyan]Adding {len(user_cols)} user features...[/cyan]")
                
                for col in user_cols:
                    feature_matrix[f'user_{col}'] = feature_matrix['userId'].map(
                        user_clean[col].to_dict()
                    ).fillna(0).astype(np.float32)
                
                # Add movie features via efficient mapping
                movie_cols = movie_clean.columns[:max_movie_features].tolist()
                self.console.print(f"[cyan]Adding {len(movie_cols)} movie features...[/cyan]")
                
                for col in movie_cols:
                    feature_matrix[f'movie_{col}'] = feature_matrix['movieId'].map(
                        movie_clean[col].to_dict()
                    ).fillna(0).astype(np.float32)
                
                # Final data preparation
                feature_cols = [col for col in feature_matrix.columns 
                            if col not in ['userId', 'movieId', 'rating']]
                
                self.console.print(f"[green]Final feature matrix: {feature_matrix.shape} with {len(feature_cols)} features[/green]")
                
                # Extract features and targets
                X = feature_matrix[feature_cols].values.astype(np.float32)
                y_reg = feature_matrix['rating'].values.astype(np.float32)
                y_clf = (feature_matrix['rating'] >= 4.0).astype(np.int32)
                
                # Validate arrays
                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    self.console.print("[yellow]Cleaning remaining NaN/inf values in features[/yellow]")
                    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Create regression dataset
                self.console.print("[cyan]Creating regression dataset...[/cyan]")
                ml_datasets['regression'] = {
                    'X': X,
                    'y': y_reg,
                    'feature_names': feature_cols,
                    'n_samples': len(X),
                    'n_features': len(feature_cols)
                }
                
                # Create classification dataset (binary: good/bad rating)
                self.console.print("[cyan]Creating classification dataset...[/cyan]")
                ml_datasets['classification'] = {
                    'X': X,  # Reuse same feature matrix
                    'y': y_clf,
                    'feature_names': feature_cols,
                    'n_samples': len(X),
                    'n_features': len(feature_cols),
                    'class_balance': {
                        'positive': float(np.sum(y_clf == 1) / len(y_clf)),
                        'negative': float(np.sum(y_clf == 0) / len(y_clf))
                    }
                }
                
                # Create placeholder clustering datasets (simplified for 4GB)
                self.console.print("[cyan]Creating simplified clustering datasets...[/cyan]")
                
                # User clustering (limited features for memory)
                user_cluster_data = user_clean.iloc[:, :15].values.astype(np.float32)
                ml_datasets['clustering_users'] = {
                    'X': user_cluster_data,
                    'feature_names': user_clean.columns[:15].tolist(),
                    'user_ids': user_clean.index.tolist(),
                    'n_samples': len(user_cluster_data),
                    'n_features': user_cluster_data.shape[1]
                }
                
                # Movie clustering (limited features for memory)  
                movie_cluster_data = movie_clean.iloc[:, :15].values.astype(np.float32)
                ml_datasets['clustering_movies'] = {
                    'X': movie_cluster_data,
                    'feature_names': movie_clean.columns[:15].tolist(),
                    'movie_ids': movie_clean.index.tolist(),
                    'n_samples': len(movie_cluster_data),
                    'n_features': movie_cluster_data.shape[1]
                }
                
                # Association rules (simple)
                high_ratings = ratings_sample[ratings_sample['rating'] >= 4.0]
                transactions = high_ratings.groupby('userId')['movieId'].apply(list).tolist()
                ml_datasets['association_rules'] = {
                    'transactions': transactions,
                    'min_support': 0.01,
                    'min_confidence': 0.5,
                    'n_transactions': len(transactions)
                }
                
                # Clean up intermediate variables
                del feature_matrix
                gc.collect()
                
                self.console.print("[green]✓ ML datasets created successfully[/green]")
                
                # Log dataset info
                for name, dataset in ml_datasets.items():
                    if dataset and 'X' in dataset:
                        self.console.print(f"[green]{name}: {dataset['X'].shape}[/green]")
                    elif dataset and 'transactions' in dataset:
                        self.console.print(f"[green]{name}: {len(dataset['transactions'])} transactions[/green]")
                
            except Exception as e:
                self.console.print(f"[red]Error creating ML datasets: {e}[/red]")
                import traceback
                traceback.print_exc()
                
                # Return None for failed datasets
                ml_datasets = {
                    'regression': None,
                    'classification': None,
                    'clustering_users': None,
                    'clustering_movies': None,
                    'association_rules': None
                }
            
            return ml_datasets
    
    def create_all_features_pipeline(self, ratings_df: pd.DataFrame, 
                                   movies_df: pd.DataFrame, 
                                   tags_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Complete feature engineering pipeline optimized for 4GB VRAM."""
        self.console.print(f"\n[bold green]Starting Optimized Feature Engineering Pipeline (4GB VRAM)[/bold green]")
        
        # Clear any existing GPU memory
        if self.use_gpu:
            self.gpu_manager.clear_cache()
        
        # Optimize data types upfront
        ratings_df['userId'] = ratings_df['userId'].astype(np.int32)
        ratings_df['movieId'] = ratings_df['movieId'].astype(np.int32)
        ratings_df['rating'] = ratings_df['rating'].astype(np.float32)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing...", total=5)
            
            # 1. User features (CPU for stability)
            progress.update(task, description="[cyan]User features...")
            user_features = self.create_user_features_optimized(ratings_df)
            progress.advance(task)
            
            # 2. Movie features
            progress.update(task, description="[cyan]Movie features...")
            movie_features = self.create_movie_features_optimized(ratings_df, movies_df)
            progress.advance(task)
            
            # 3. ML datasets (regression & classification only)
            progress.update(task, description="[cyan]ML datasets...")
            ml_datasets = self.create_ml_datasets(ratings_df, user_features, movie_features)
            progress.advance(task)
            
            # 4. Sparse matrix (simplified)
            progress.update(task, description="[cyan]Sparse matrix...")
            user_item_matrix, user_mapping, movie_mapping = self.create_user_item_matrix_simple(ratings_df)
            progress.advance(task)
            
            # 5. Final cleanup
            progress.update(task, description="[cyan]Finalizing...")
            gc.collect()
            if self.use_gpu:
                self.gpu_manager.clear_cache()
            progress.advance(task)
        
        return {
            'user_features': user_features,
            'movie_features': movie_features,
            'ml_datasets': ml_datasets,
            'user_item_matrix': user_item_matrix,
            'user_mapping': user_mapping,
            'movie_mapping': movie_mapping
        }
    
    def create_user_item_matrix_simple(self, ratings_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Create user-item matrix using simple CPU approach."""
        self.console.print("[cyan]Creating sparse user-item matrix...[/cyan]")
        
        # Use pandas categories for efficient mapping
        ratings_df['userId'] = ratings_df['userId'].astype('category')
        ratings_df['movieId'] = ratings_df['movieId'].astype('category')
        
        user_codes = ratings_df['userId'].cat.codes.values
        movie_codes = ratings_df['movieId'].cat.codes.values
        
        # Create mappings
        user_mapping = dict(enumerate(ratings_df['userId'].cat.categories))
        movie_mapping = dict(enumerate(ratings_df['movieId'].cat.categories))
        
        # Create sparse matrix
        matrix = csr_matrix(
            (ratings_df['rating'].values, (user_codes, movie_codes)),
            shape=(len(user_mapping), len(movie_mapping)),
            dtype=np.float32
        )
        
        self.console.print(f"[green]✓ Created {matrix.shape} sparse matrix[/green]")
        return matrix, user_mapping, movie_mapping
    
    def save_features(self, features_dict: Dict, output_dir: str = "data/processed"):
        """Save features efficiently."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save regular features
        for name, data in features_dict.items():
            if name == 'ml_datasets':
                # Save ML datasets separately
                output_path = os.path.join(output_dir, "ml_ready_datasets.pkl.gz")
                import gzip
                with gzip.open(output_path, 'wb') as f:
                    pickle.dump(data, f, protocol=4)
                self.console.print(f"[green]✓ Saved ML datasets to {output_path}[/green]")
            elif name == 'user_item_matrix':
                # Save sparse matrix
                output_path = os.path.join(output_dir, "user_item_matrix.npz")
                sparse.save_npz(output_path, data)
                self.console.print(f"[green]✓ Saved sparse matrix to {output_path}[/green]")
            elif isinstance(data, pd.DataFrame):
                # Save DataFrames as parquet
                output_path = os.path.join(output_dir, f"{name}.parquet")
                data.to_parquet(output_path, compression='snappy')
                self.console.print(f"[green]✓ Saved {name} to {output_path}[/green]")

# Alias for compatibility
GPUHyperOptimizedDataTransformer = HyperOptimizedDataTransformer