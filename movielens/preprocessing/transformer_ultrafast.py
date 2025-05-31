import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix, coo_matrix, hstack
from scipy import sparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import gc
import numba
from numba import jit, prange, njit
import swifter
from functools import lru_cache
import pickle
from sklearn.utils import murmurhash3_32

# GPU imports with fallbacks
try:
    import cupy as cp
    import cuml
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    from cuml.preprocessing import MinMaxScaler as CuMinMaxScaler
    from cuml.decomposition import PCA as CuPCA
    from cuml.decomposition import TruncatedSVD as CuTruncatedSVD
    from cuml.cluster import KMeans as CuKMeans
    from cuml.manifold import TSNE as CuTSNE
    from cuml.neighbors import NearestNeighbors as CuNearestNeighbors
    from cuml.metrics import pairwise_distances as cu_pairwise_distances
    from cuml.common import logger as cuml_logger
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
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            try:
                # Initialize RMM memory pool for better GPU memory management
                rmm.reinitialize(pool_allocator=True)
                self.memory_info = cp.cuda.runtime.memGetInfo()
                self.total_memory = self.memory_info[1]
                self.console = Console()
                self.console.print(f"[green]✓ GPU Memory Pool initialized: {self.total_memory / 1024**3:.1f}GB total[/green]")
            except Exception as e:
                self.gpu_available = False
                print(f"[yellow]GPU memory setup failed: {e}[/yellow]")
    
    def get_available_memory(self):
        """Get available GPU memory in bytes."""
        if not self.gpu_available:
            return 0
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return free_mem
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
    
    def can_fit_array(self, shape, dtype=cp.float32):
        """Check if array of given shape can fit in GPU memory."""
        if not self.gpu_available:
            return False
        required_bytes = np.prod(shape) * cp.dtype(dtype).itemsize
        available_bytes = self.get_available_memory()
        return required_bytes < (available_bytes * 0.8)  # Use 80% of available memory

# Enhanced numba functions for maximum speed (CPU fallback)
@njit(parallel=True, cache=True, fastmath=True)
def ultra_fast_groupby_stats(user_ids, ratings, movie_ids, timestamps, n_users):
    """Ultra-fast groupby with all statistics in one pass."""
    # Pre-allocate all statistics arrays
    counts = np.zeros(n_users, dtype=np.int32)
    sums = np.zeros(n_users, dtype=np.float32)
    sum_squares = np.zeros(n_users, dtype=np.float32)
    mins = np.full(n_users, 5.0, dtype=np.float32)
    maxs = np.zeros(n_users, dtype=np.float32)
    first_time = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
    last_time = np.zeros(n_users, dtype=np.int64)
    unique_movies = np.zeros((n_users, 1000), dtype=np.int32)  # Track up to 1000 movies per user
    unique_counts = np.zeros(n_users, dtype=np.int32)
    
    # Single pass through data
    for i in prange(len(user_ids)):
        uid = user_ids[i]
        rating = ratings[i]
        movie = movie_ids[i]
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
            
        # Track unique movies (approximate)
        if unique_counts[uid] < 1000:
            unique_movies[uid, unique_counts[uid]] = movie
            unique_counts[uid] += 1
    
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

def gpu_groupby_stats(user_ids, ratings, movie_ids, timestamps, n_users):
    """GPU-accelerated groupby statistics using CuPy."""
    if not GPU_AVAILABLE:
        return ultra_fast_groupby_stats(user_ids, ratings, movie_ids, timestamps, n_users)
    
    # Transfer data to GPU
    user_ids_gpu = cp.asarray(user_ids, dtype=cp.int32)
    ratings_gpu = cp.asarray(ratings, dtype=cp.float32)
    movie_ids_gpu = cp.asarray(movie_ids, dtype=cp.int32)
    timestamps_gpu = cp.asarray(timestamps, dtype=cp.int64)
    
    # Pre-allocate GPU arrays
    counts = cp.zeros(n_users, dtype=cp.int32)
    sums = cp.zeros(n_users, dtype=cp.float32)
    sum_squares = cp.zeros(n_users, dtype=cp.float32)
    mins = cp.full(n_users, 5.0, dtype=cp.float32)
    maxs = cp.zeros(n_users, dtype=cp.float32)
    first_time = cp.full(n_users, cp.iinfo(cp.int64).max, dtype=cp.int64)
    last_time = cp.zeros(n_users, dtype=cp.int64)
    unique_counts = cp.zeros(n_users, dtype=cp.int32)
    
    # GPU kernel for statistics computation
    kernel = cp.ElementwiseKernel(
        'int32 uid, float32 rating, int32 movie, int64 timestamp',
        'raw int32 counts, raw float32 sums, raw float32 sum_squares, '
        'raw float32 mins, raw float32 maxs, raw int64 first_time, raw int64 last_time',
        '''
        atomicAdd(&counts[uid], 1);
        atomicAdd(&sums[uid], rating);
        atomicAdd(&sum_squares[uid], rating * rating);
        atomicMin(&mins[uid], rating);
        atomicMax(&maxs[uid], rating);
        atomicMin(&first_time[uid], timestamp);
        atomicMax(&last_time[uid], timestamp);
        ''',
        'gpu_stats_kernel'
    )
    
    # Execute kernel
    kernel(user_ids_gpu, ratings_gpu, movie_ids_gpu, timestamps_gpu,
           counts, sums, sum_squares, mins, maxs, first_time, last_time)
    
    # Calculate derived statistics on GPU
    means = cp.divide(sums, cp.maximum(counts, 1))
    variances = cp.divide(sum_squares, cp.maximum(counts, 1)) - means**2
    stds = cp.sqrt(cp.maximum(variances, 0))
    
    # Transfer results back to CPU
    return (cp.asnumpy(counts), cp.asnumpy(means), cp.asnumpy(stds), 
            cp.asnumpy(mins), cp.asnumpy(maxs), cp.asnumpy(first_time), 
            cp.asnumpy(last_time), cp.asnumpy(unique_counts))

@njit(parallel=True, cache=True)
def fast_cluster_assignment(data, centroids):
    """Ultra-fast cluster assignment using Euclidean distance."""
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    n_features = data.shape[1]
    labels = np.zeros(n_samples, dtype=np.int32)
    
    for i in prange(n_samples):
        min_dist = np.inf
        best_cluster = 0
        
        for j in range(n_clusters):
            dist = 0.0
            for k in range(n_features):
                diff = data[i, k] - centroids[j, k]
                dist += diff * diff
            
            if dist < min_dist:
                min_dist = dist
                best_cluster = j
        
        labels[i] = best_cluster
    
    return labels

class GPUHyperOptimizedDataTransformer:
    """GPU-accelerated hyper-optimized transformer using CuPy and cuML."""
    
    def __init__(self, n_jobs: int = None, use_gpu: bool = True):
        self.console = Console()
        self.n_jobs = n_jobs if n_jobs else mp.cpu_count()
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize GPU memory manager
        self.gpu_manager = GPUMemoryManager()
        self.use_gpu = self.use_gpu and self.gpu_manager.gpu_available
        
        # Pre-initialize transformers based on GPU availability
        if self.use_gpu:
            self.scalers = {
                'standard': CuStandardScaler(),
                'minmax': CuMinMaxScaler()
            }
            self.pca_models = {}
            self.svd_models = {}
            self.kmeans_models = {}
        else:
            self.scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler()
            }
            self.pca_models = {}
            self.svd_models = {}
            self.kmeans_models = {}
        
        self.encoders = {}
        self.transformation_cache = {}
        
        # Pre-compile numba functions
        self._precompile_numba()
        
        pd.options.mode.chained_assignment = None
        
        gpu_status = "GPU-ACCELERATED" if self.use_gpu else "CPU-ONLY"
        self.console.print(f"[green]✓ {gpu_status} Hyper-Optimized Transformer initialized with {self.n_jobs} cores[/green]")
        
        if self.use_gpu:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            self.console.print(f"[blue]GPU Memory: {free_mem / 1024**3:.1f}GB free / {total_mem / 1024**3:.1f}GB total[/blue]")
    
    def _precompile_numba(self):
        """Pre-compile numba functions for faster first run."""
        dummy_data = np.array([0], dtype=np.int32)
        dummy_ratings = np.array([3.5], dtype=np.float32)
        try:
            ultra_fast_groupby_stats(dummy_data, dummy_ratings, dummy_data, dummy_data, 1)
        except:
            pass
    
    def _to_gpu(self, data):
        """Transfer data to GPU if available and beneficial."""
        if not self.use_gpu:
            return data
        
        if isinstance(data, pd.DataFrame):
            if self.gpu_manager.can_fit_array(data.shape, cp.float32):
                return cp.asarray(data.values, dtype=cp.float32)
        elif isinstance(data, np.ndarray):
            if self.gpu_manager.can_fit_array(data.shape, data.dtype):
                return cp.asarray(data)
        
        return data
    
    def _to_cpu(self, data):
        """Transfer data back to CPU."""
        if hasattr(data, 'get'):  # CuPy array
            return data.get()
        return data
    
    def create_user_features_gpu_accelerated(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated user feature creation."""
        self.console.print("[cyan]Creating user features (GPU-ACCELERATED mode)...[/cyan]")
        
        # Convert to numpy arrays
        unique_users = ratings_df['userId'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        
        # Vectorized conversion to arrays
        user_indices = np.fromiter((user_to_idx[uid] for uid in ratings_df['userId'].values), 
                                   dtype=np.int32, count=len(ratings_df))
        ratings = ratings_df['rating'].values.astype(np.float32)
        movie_ids = ratings_df['movieId'].values.astype(np.int32)
        timestamps = ratings_df['timestamp'].astype(np.int64).values // 10**9
        
        # Use GPU-accelerated or CPU-optimized function
        if self.use_gpu and len(ratings_df) > 100000:
            self.console.print("[blue]Using GPU acceleration for groupby statistics...[/blue]")
            counts, means, stds, mins, maxs, first_time, last_time, unique_counts = \
                gpu_groupby_stats(user_indices, ratings, movie_ids, timestamps, len(unique_users))
        else:
            counts, means, stds, mins, maxs, first_time, last_time, unique_counts = \
                ultra_fast_groupby_stats(user_indices, ratings, movie_ids, timestamps, len(unique_users))
        
        # Create DataFrame in one shot
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
        
        # GPU-accelerated vectorized operations if data fits in GPU memory
        if self.use_gpu and self.gpu_manager.can_fit_array(user_features.shape):
            self.console.print("[blue]Using GPU for derived feature calculations...[/blue]")
            
            # Transfer to GPU
            first_time_gpu = cp.asarray(first_time)
            last_time_gpu = cp.asarray(last_time)
            rating_max_gpu = cp.asarray(maxs)
            rating_min_gpu = cp.asarray(mins)
            rating_count_gpu = cp.asarray(counts)
            unique_counts_gpu = cp.asarray(unique_counts)
            
            # GPU calculations
            time_span_days = (last_time_gpu - first_time_gpu) / 86400
            rating_range = rating_max_gpu - rating_min_gpu
            rating_frequency = rating_count_gpu / (time_span_days + 1)
            
            # Quantile calculations on GPU
            q75_count = cp.percentile(rating_count_gpu, 75)
            q75_diversity = cp.percentile(unique_counts_gpu, 75)
            
            is_active = (rating_count_gpu > q75_count).astype(cp.int8)
            is_diverse = (unique_counts_gpu > q75_diversity).astype(cp.int8)
            
            # Transfer results back to CPU
            user_features['time_span_days'] = cp.asnumpy(time_span_days)
            user_features['rating_range'] = cp.asnumpy(rating_range)
            user_features['rating_frequency'] = cp.asnumpy(rating_frequency)
            user_features['is_active'] = cp.asnumpy(is_active)
            user_features['is_diverse'] = cp.asnumpy(is_diverse)
            
            # Clear GPU memory
            self.gpu_manager.clear_cache()
        else:
            # CPU fallback
            user_features['time_span_days'] = (user_features['last_timestamp'] - user_features['first_timestamp']) / 86400
            user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
            user_features['rating_frequency'] = user_features['rating_count'] / (user_features['time_span_days'] + 1)
            
            q75_count = np.percentile(counts, 75)
            q75_diversity = np.percentile(unique_counts, 75)
            
            user_features['is_active'] = (user_features['rating_count'] > q75_count).astype(np.int8)
            user_features['is_diverse'] = (user_features['movie_diversity'] > q75_diversity).astype(np.int8)
        
        # Drop temporary columns
        user_features = user_features.drop(['first_timestamp', 'last_timestamp'], axis=1)
        
        # Optimize dtypes
        float_cols = user_features.select_dtypes(include=['float64']).columns
        user_features[float_cols] = user_features[float_cols].astype(np.float32)
        
        gc.collect()
        
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features with GPU acceleration[/green]")
        return user_features
    
    def create_movie_features_gpu_accelerated(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated movie feature creation."""
        self.console.print("[cyan]Creating movie features (GPU-ACCELERATED mode)...[/cyan]")
        
        # Use GPU for large datasets
        if self.use_gpu and len(ratings_df) > 50000:
            self.console.print("[blue]Using GPU for movie statistics computation...[/blue]")
            
            # Transfer to GPU
            movie_ids = cp.asarray(ratings_df['movieId'].values)
            ratings = cp.asarray(ratings_df['rating'].values)
            user_ids = cp.asarray(ratings_df['userId'].values)
            
            unique_movies = cp.unique(movie_ids)
            n_movies = len(unique_movies)
            
            # GPU-based aggregation using advanced indexing
            movie_counts = cp.zeros(n_movies, dtype=cp.int32)
            movie_sums = cp.zeros(n_movies, dtype=cp.float32)
            movie_sum_squares = cp.zeros(n_movies, dtype=cp.float32)
            
            # Create mapping for GPU operations
            movie_to_idx = cp.arange(len(unique_movies))
            movie_idx_map = cp.zeros(movie_ids.max() + 1, dtype=cp.int32) - 1
            movie_idx_map[unique_movies] = movie_to_idx
            
            # Map movie IDs to indices
            movie_indices = movie_idx_map[movie_ids]
            valid_mask = movie_indices >= 0
            
            # GPU aggregation
            cp.add.at(movie_counts, movie_indices[valid_mask], 1)
            cp.add.at(movie_sums, movie_indices[valid_mask], ratings[valid_mask])
            cp.add.at(movie_sum_squares, movie_indices[valid_mask], ratings[valid_mask]**2)
            
            # Calculate statistics on GPU
            movie_means = movie_sums / cp.maximum(movie_counts, 1)
            movie_vars = (movie_sum_squares / cp.maximum(movie_counts, 1)) - movie_means**2
            movie_stds = cp.sqrt(cp.maximum(movie_vars, 0))
            
            # Calculate user diversity on GPU
            user_diversity = cp.zeros(n_movies, dtype=cp.int32)
            for i in range(n_movies):
                mask = movie_indices == i
                if mask.any():
                    user_diversity[i] = len(cp.unique(user_ids[mask]))
            
            # Transfer back to CPU
            movie_features = pd.DataFrame({
                'rating_count': cp.asnumpy(movie_counts),
                'rating_mean': cp.asnumpy(movie_means),
                'rating_std': cp.asnumpy(movie_stds),
                'user_diversity': cp.asnumpy(user_diversity)
            }, index=cp.asnumpy(unique_movies))
            
            # Clear GPU memory
            self.gpu_manager.clear_cache()
            
        else:
            # CPU fallback for smaller datasets
            unique_movies = ratings_df['movieId'].unique()
            movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
            
            n_movies = len(unique_movies)
            movie_counts = np.zeros(n_movies, dtype=np.int32)
            movie_sums = np.zeros(n_movies, dtype=np.float32)
            movie_sum_squares = np.zeros(n_movies, dtype=np.float32)
            movie_user_sets = [set() for _ in range(n_movies)]
            
            # Single pass through data
            for _, row in ratings_df.iterrows():
                idx = movie_to_idx[row['movieId']]
                rating = row['rating']
                movie_counts[idx] += 1
                movie_sums[idx] += rating
                movie_sum_squares[idx] += rating * rating
                movie_user_sets[idx].add(row['userId'])
            
            # Calculate statistics
            movie_means = movie_sums / np.maximum(movie_counts, 1)
            movie_vars = (movie_sum_squares / np.maximum(movie_counts, 1)) - movie_means**2
            movie_stds = np.sqrt(np.maximum(movie_vars, 0))
            user_diversity = np.array([len(s) for s in movie_user_sets])
            
            movie_features = pd.DataFrame({
                'rating_count': movie_counts,
                'rating_mean': movie_means,
                'rating_std': movie_stds,
                'user_diversity': user_diversity
            }, index=unique_movies)
        
        # Merge with movie metadata
        movie_features = movie_features.join(movies_df.set_index('movieId'), how='left')
        
        # GPU-accelerated genre encoding using hashing
        if 'genre_list' in movie_features.columns:
            n_genres = 50
            
            if self.use_gpu and len(movie_features) > 10000:
                self.console.print("[blue]Using GPU for genre encoding...[/blue]")
                
                # GPU-based hashing
                genre_matrix = cp.zeros((len(movie_features), n_genres), dtype=cp.float32)
                
                for idx, genres in enumerate(movie_features['genre_list'].fillna('')):
                    if isinstance(genres, list):
                        for genre in genres:
                            hash_idx = murmurhash3_32(genre, positive=True) % n_genres
                            genre_matrix[idx, hash_idx] = 1
                
                # Add genre features
                genre_df = pd.DataFrame(
                    cp.asnumpy(genre_matrix),
                    columns=[f'genre_hash_{i}' for i in range(n_genres)],
                    index=movie_features.index
                )
                movie_features = pd.concat([movie_features, genre_df], axis=1)
                movie_features['genre_count'] = cp.asnumpy(genre_matrix.sum(axis=1))
                
                self.gpu_manager.clear_cache()
            else:
                # CPU fallback
                genre_matrix = np.zeros((len(movie_features), n_genres), dtype=np.float32)
                
                for idx, genres in enumerate(movie_features['genre_list'].fillna('')):
                    if isinstance(genres, list):
                        for genre in genres:
                            hash_idx = murmurhash3_32(genre, positive=True) % n_genres
                            genre_matrix[idx, hash_idx] = 1
                
                genre_df = pd.DataFrame(
                    genre_matrix,
                    columns=[f'genre_hash_{i}' for i in range(n_genres)],
                    index=movie_features.index
                )
                movie_features = pd.concat([movie_features, genre_df], axis=1)
                movie_features['genre_count'] = genre_matrix.sum(axis=1)
        
        # Vectorized derived features
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
        
        self.console.print(f"[green]✓ Created {len(movie_features.columns)} movie features with GPU acceleration[/green]")
        return movie_features
    
    def create_cold_start_features_gpu_accelerated(self, ratings_df: pd.DataFrame, 
                                                  user_features: pd.DataFrame,
                                                  movie_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """GPU-accelerated cold start feature creation."""
        self.console.print("[cyan]Creating cold start features (GPU-ACCELERATED)...[/cyan]")
        
        # User cold start features
        user_cold_features = pd.DataFrame(index=user_features.index)
        
        if self.use_gpu and len(user_features) > 1000:
            self.console.print("[blue]Using GPU for cold start feature computation...[/blue]")
            
            # Transfer to GPU
            rating_counts_gpu = cp.asarray(user_features['rating_count'].values)
            
            # GPU-based cold start detection
            cold_threshold = cp.percentile(rating_counts_gpu, 10)
            is_cold_start = (rating_counts_gpu < cold_threshold).astype(cp.int8)
            rating_confidence = 1 - cp.exp(-rating_counts_gpu / 10)
            
            # Transfer back to CPU
            user_cold_features['is_cold_start'] = cp.asnumpy(is_cold_start)
            user_cold_features['rating_confidence'] = cp.asnumpy(rating_confidence)
            
        else:
            # CPU fallback
            cold_threshold = np.percentile(user_features['rating_count'].values, 10)
            user_cold_features['is_cold_start'] = (user_features['rating_count'] < cold_threshold).astype(np.int8)
            user_cold_features['rating_confidence'] = 1 - np.exp(-user_features['rating_count'].values / 10)
        
        # GPU-accelerated clustering
        if len(user_features) > 1000:
            key_features = ['rating_count', 'rating_mean', 'rating_std', 'movie_diversity']
            X_user = user_features[key_features].fillna(0).values
            
            if self.use_gpu and self.gpu_manager.can_fit_array(X_user.shape):
                self.console.print("[blue]Using cuML for user clustering...[/blue]")
                
                # GPU-based preprocessing and clustering
                scaler = CuStandardScaler()
                X_user_gpu = cp.asarray(X_user, dtype=cp.float32)
                X_user_scaled = scaler.fit_transform(X_user_gpu)
                
                # GPU SVD
                n_components = min(10, X_user_scaled.shape[1])
                svd = CuTruncatedSVD(n_components=n_components, random_state=42)
                X_user_reduced = svd.fit_transform(X_user_scaled)
                
                # GPU K-means
                n_clusters = min(20, len(user_features) // 500)
                kmeans = CuKMeans(
                    n_clusters=n_clusters,
                    max_iter=10,
                    random_state=42
                )
                
                user_clusters = kmeans.fit_predict(X_user_reduced)
                user_cold_features['user_cluster'] = cp.asnumpy(user_clusters)
                
                # GPU-based cluster statistics
                n_clusters_actual = len(cp.unique(user_clusters))
                cluster_ratings = cp.zeros(n_clusters_actual)
                cluster_counts = cp.zeros(n_clusters_actual)
                
                # Vectorized aggregation on GPU
                ratings_gpu = cp.asarray(ratings_df['rating'].values)
                user_ids_gpu = cp.asarray(ratings_df['userId'].values)
                
                for cluster_id in range(n_clusters_actual):
                    cluster_mask = user_clusters == cluster_id
                    cluster_users = cp.asarray(user_features.index[cp.asnumpy(cluster_mask)])
                    
                    if len(cluster_users) > 0:
                        user_mask = cp.isin(user_ids_gpu, cluster_users)
                        if user_mask.any():
                            cluster_ratings[cluster_id] = ratings_gpu[user_mask].sum()
                            cluster_counts[cluster_id] = user_mask.sum()
                
                cluster_means = cluster_ratings / cp.maximum(cluster_counts, 1)
                user_cold_features['cluster_avg_rating'] = user_cold_features['user_cluster'].map(
                    dict(enumerate(cp.asnumpy(cluster_means)))
                ).fillna(3.5)
                
                self.gpu_manager.clear_cache()
                
            else:
                # CPU fallback clustering
                X_user_scaled = (X_user - X_user.mean(axis=0)) / (X_user.std(axis=0) + 1e-8)
                
                n_components = min(10, X_user_scaled.shape[1])
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                X_user_reduced = svd.fit_transform(X_user_scaled)
                
                n_clusters = min(20, len(user_features) // 500)
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(20000, len(user_features)),
                    max_iter=10,
                    n_init=1,
                    random_state=42,
                    reassignment_ratio=0.001
                )
                
                user_cold_features['user_cluster'] = kmeans.fit_predict(X_user_reduced)
                
                # Cluster statistics
                cluster_ratings = np.zeros(n_clusters)
                cluster_counts = np.zeros(n_clusters)
                
                user_cluster_map = dict(zip(user_features.index, user_cold_features['user_cluster']))
                
                for user_id, cluster in user_cluster_map.items():
                    user_mask = ratings_df['userId'] == user_id
                    if user_mask.any():
                        cluster_ratings[cluster] += ratings_df.loc[user_mask, 'rating'].sum()
                        cluster_counts[cluster] += user_mask.sum()
                
                cluster_means = cluster_ratings / np.maximum(cluster_counts, 1)
                user_cold_features['cluster_avg_rating'] = user_cold_features['user_cluster'].map(
                    dict(enumerate(cluster_means))
                ).fillna(3.5)
        
        # Movie cold start features (similar GPU acceleration)
        movie_cold_features = pd.DataFrame(index=movie_features.index)
        
        if self.use_gpu and len(movie_features) > 1000:
            rating_counts_gpu = cp.asarray(movie_features['rating_count'].values)
            movie_cold_threshold = cp.percentile(rating_counts_gpu, 10)
            movie_cold_features['is_cold_start'] = cp.asnumpy((rating_counts_gpu < movie_cold_threshold).astype(cp.int8))
            movie_cold_features['rating_confidence'] = cp.asnumpy(1 - cp.exp(-rating_counts_gpu / 10))
        else:
            movie_cold_threshold = np.percentile(movie_features['rating_count'].values, 10)
            movie_cold_features['is_cold_start'] = (movie_features['rating_count'] < movie_cold_threshold).astype(np.int8)
            movie_cold_features['rating_confidence'] = 1 - np.exp(-movie_features['rating_count'].values / 10)
        
        # GPU-accelerated movie clustering using genre features
        if len(movie_features) > 1000:
            genre_cols = [col for col in movie_features.columns if col.startswith('genre_hash_')]
            if len(genre_cols) >= 5:
                X_movie = movie_features[genre_cols[:20]].values
                
                if self.use_gpu and self.gpu_manager.can_fit_array(X_movie.shape):
                    self.console.print("[blue]Using cuML for movie clustering...[/blue]")
                    
                    X_movie_gpu = cp.asarray(X_movie, dtype=cp.float32)
                    n_clusters = min(50, len(movie_features) // 200)
                    
                    movie_kmeans = CuKMeans(
                        n_clusters=n_clusters,
                        max_iter=10,
                        random_state=42
                    )
                    
                    movie_clusters = movie_kmeans.fit_predict(X_movie_gpu)
                    movie_cold_features['movie_cluster'] = cp.asnumpy(movie_clusters)
                    
                    self.gpu_manager.clear_cache()
                else:
                    # CPU fallback
                    n_clusters = min(50, len(movie_features) // 200)
                    movie_kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=min(10000, len(movie_features)),
                        max_iter=10,
                        n_init=1,
                        random_state=42
                    )
                    
                    movie_cold_features['movie_cluster'] = movie_kmeans.fit_predict(X_movie)
        
        # Global averages
        global_avg = np.float32(3.5)
        user_cold_features['global_avg_rating'] = global_avg
        movie_cold_features['global_avg_rating'] = global_avg
        
        gc.collect()
        
        return user_cold_features, movie_cold_features
    
    def create_all_features_pipeline_gpu(self, ratings_df: pd.DataFrame, 
                                        movies_df: pd.DataFrame, 
                                        tags_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Complete GPU-accelerated feature engineering pipeline."""
        gpu_status = "GPU-ACCELERATED" if self.use_gpu else "CPU-OPTIMIZED"
        self.console.print(f"\n[bold green]Starting {gpu_status} Feature Engineering Pipeline[/bold green]")
        
        if self.use_gpu:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            self.console.print(f"[blue]GPU Memory Available: {free_mem / 1024**3:.1f}GB / {total_mem / 1024**3:.1f}GB[/blue]")
        
        # Pre-allocate memory and optimize data types upfront
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
            task = progress.add_task(f"[cyan]{gpu_status} Processing...", total=6)
            
            # 1. Temporal features (vectorized)
            progress.update(task, description="[cyan]Temporal features...")
            ratings_df['hour'] = ratings_df['timestamp'].dt.hour.astype(np.int8)
            ratings_df['day_of_week'] = ratings_df['timestamp'].dt.dayofweek.astype(np.int8)
            ratings_df['is_weekend'] = (ratings_df['day_of_week'] >= 5).astype(np.int8)
            progress.advance(task)
            
            # 2. User features (GPU-accelerated)
            progress.update(task, description="[cyan]GPU User features...")
            user_features = self.create_user_features_gpu_accelerated(ratings_df)
            progress.advance(task)
            
            # 3. Movie features (GPU-accelerated)
            progress.update(task, description="[cyan]GPU Movie features...")
            movie_features = self.create_movie_features_gpu_accelerated(ratings_df, movies_df)
            progress.advance(task)
            
            # 4. Cold start features (GPU-accelerated)
            progress.update(task, description="[cyan]GPU Cold start features...")
            user_cold, movie_cold = self.create_cold_start_features_gpu_accelerated(
                ratings_df, user_features, movie_features
            )
            user_features = pd.concat([user_features, user_cold], axis=1)
            movie_features = pd.concat([movie_features, movie_cold], axis=1)
            progress.advance(task)
            
            # 5. Sparse matrix creation (GPU-accelerated)
            progress.update(task, description="[cyan]GPU Sparse matrices...")
            user_item_matrix, user_mapping, movie_mapping = self.create_user_item_matrix_gpu_accelerated(ratings_df)
            progress.advance(task)
            
            # 6. Final optimization
            progress.update(task, description="[cyan]Final optimization...")
            # Convert all float64 to float32
            for df in [user_features, movie_features]:
                float_cols = df.select_dtypes(include=['float64']).columns
                df[float_cols] = df[float_cols].astype(np.float32)
            
            if self.use_gpu:
                self.gpu_manager.clear_cache()
            gc.collect()
            progress.advance(task)
        
        # Enhanced ratings dataframe with all temporal features
        ratings_enhanced = ratings_df.copy()
        
        return {
            'ratings_enhanced': ratings_enhanced,
            'user_features': user_features,
            'movie_features': movie_features,
            'user_item_matrix': user_item_matrix,
            'user_mapping': user_mapping,
            'movie_mapping': movie_mapping
        }
    
    def create_user_item_matrix_gpu_accelerated(self, ratings_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """GPU-accelerated user-item matrix creation using optimized sparse operations."""
        
        if self.use_gpu and len(ratings_df) > 100000:
            self.console.print("[blue]Using GPU for sparse matrix creation...[/blue]")
            
            # Transfer to GPU
            user_ids = cp.asarray(ratings_df['userId'].values)
            movie_ids = cp.asarray(ratings_df['movieId'].values)
            ratings = cp.asarray(ratings_df['rating'].values)
            
            # Get unique IDs on GPU
            unique_users = cp.unique(user_ids)
            unique_movies = cp.unique(movie_ids)
            
            # Create mappings
            user_mapping = dict(enumerate(cp.asnumpy(unique_users)))
            movie_mapping = dict(enumerate(cp.asnumpy(unique_movies)))
            
            # Create index mapping arrays on GPU
            user_idx_map = cp.zeros(user_ids.max() + 1, dtype=cp.int32) - 1
            movie_idx_map = cp.zeros(movie_ids.max() + 1, dtype=cp.int32) - 1
            
            user_idx_map[unique_users] = cp.arange(len(unique_users))
            movie_idx_map[unique_movies] = cp.arange(len(unique_movies))
            
            # Map to indices
            user_indices = user_idx_map[user_ids]
            movie_indices = movie_idx_map[movie_ids]
            
            # Transfer back to CPU for scipy sparse matrix creation
            user_indices_cpu = cp.asnumpy(user_indices)
            movie_indices_cpu = cp.asnumpy(movie_indices)
            ratings_cpu = cp.asnumpy(ratings)
            
            # Create sparse matrix
            matrix = csr_matrix(
                (ratings_cpu, (user_indices_cpu, movie_indices_cpu)),
                shape=(len(user_mapping), len(movie_mapping)),
                dtype=np.float32
            )
            
            self.gpu_manager.clear_cache()
            
        else:
            # CPU fallback with category codes
            ratings_df['userId'] = ratings_df['userId'].astype('category')
            ratings_df['movieId'] = ratings_df['movieId'].astype('category')
            
            user_codes = ratings_df['userId'].cat.codes.values
            movie_codes = ratings_df['movieId'].cat.codes.values
            
            # Create mappings
            user_mapping = dict(enumerate(ratings_df['userId'].cat.categories))
            movie_mapping = dict(enumerate(ratings_df['movieId'].cat.categories))
            
            # Create sparse matrix in one shot
            matrix = csr_matrix(
                (ratings_df['rating'].values, (user_codes, movie_codes)),
                shape=(len(user_mapping), len(movie_mapping)),
                dtype=np.float32
            )
        
        return matrix, user_mapping, movie_mapping
    
    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'gpu_manager') and self.gpu_manager.gpu_available:
            self.gpu_manager.clear_cache()

# Alias for backward compatibility
HyperOptimizedDataTransformer = GPUHyperOptimizedDataTransformer