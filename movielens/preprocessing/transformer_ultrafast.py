import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import gc
import numba
from numba import jit, prange
import swifter 
from functools import lru_cache
import pickle

warnings.filterwarnings('ignore')

# Enable numba JIT compilation for critical functions
@jit(nopython=True, parallel=True, cache=True)
def fast_groupby_stats(user_ids, ratings, movie_ids, n_users):
    """Ultra-fast groupby statistics using numba."""
    user_stats = np.zeros((n_users, 8), dtype=np.float32)
    
    for i in prange(len(user_ids)):
        uid = user_ids[i]
        user_stats[uid, 0] += 1  # count
        user_stats[uid, 1] += ratings[i]  # sum for mean
        user_stats[uid, 7] = movie_ids[i]  # track last movie (for unique count approximation)
    
    # Calculate derived stats
    for i in prange(n_users):
        if user_stats[i, 0] > 0:
            user_stats[i, 2] = user_stats[i, 1] / user_stats[i, 0]  # mean
    
    return user_stats

class UltraFastDataTransformer:
    """Ultra-optimized data transformer for systems with 32GB+ RAM."""
    
    def __init__(self, n_jobs: int = None):
        self.console = Console()
        self.n_jobs = n_jobs if n_jobs else mp.cpu_count()
        self.scalers = {}
        self.encoders = {}
        self.transformation_cache = {}
        
        # Initialize swifter for pandas acceleration
        pd.options.mode.chained_assignment = None
        
        self.console.print(f"[green]✓ UltraFast Transformer initialized with {self.n_jobs} cores[/green]")
    
    def optimize_dtypes_ultra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast dtype optimization without memory constraints."""
        # Use float32 instead of float64 for all floats
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        
        # Use int32 for most integers (faster than smaller types on modern CPUs)
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if df[col].max() < 2147483647:
                df[col] = df[col].astype(np.int32)
        
        # Don't use categories for IDs - keep as integers for faster operations
        return df
    
    def create_user_features_ultra_fast(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast user feature creation using vectorized operations and numba."""
        self.console.print("[cyan]Creating user features (ULTRA-FAST mode)...[/cyan]")
        
        # Pre-compute mappings
        unique_users = ratings_df['userId'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        
        # Convert to numpy arrays for faster processing
        user_indices = ratings_df['userId'].map(user_to_idx).values.astype(np.int32)
        ratings = ratings_df['rating'].values.astype(np.float32)
        movie_ids = ratings_df['movieId'].values.astype(np.int32)
        timestamps = ratings_df['timestamp'].values
        
        # Use numba-accelerated function for basic stats
        user_stats = fast_groupby_stats(user_indices, ratings, movie_ids, len(unique_users))
        
        # Create base features dataframe
        user_features = pd.DataFrame(
            user_stats[:, :3],
            columns=['rating_count', 'rating_sum', 'rating_mean'],
            index=unique_users
        )
        
        # Vectorized operations for additional features
        user_groups = ratings_df.groupby('userId', sort=False)
        
        # Use parallel aggregation
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Split into chunks for parallel processing
            chunk_size = len(unique_users) // self.n_jobs
            user_chunks = [unique_users[i:i+chunk_size] for i in range(0, len(unique_users), chunk_size)]
            
            def process_user_chunk(users):
                chunk_features = {}
                for user in users:
                    user_data = ratings_df[ratings_df['userId'] == user]
                    if len(user_data) > 0:
                        chunk_features[user] = {
                            'rating_std': user_data['rating'].std(),
                            'rating_min': user_data['rating'].min(),
                            'rating_max': user_data['rating'].max(),
                            'movie_diversity': user_data['movieId'].nunique(),
                            'time_span': (user_data['timestamp'].max() - user_data['timestamp'].min()).total_seconds() / 86400
                        }
                return chunk_features
            
            # Process chunks in parallel
            futures = [executor.submit(process_user_chunk, chunk) for chunk in user_chunks]
            
            # Combine results
            all_features = {}
            for future in futures:
                all_features.update(future.result())
        
        # Convert to DataFrame
        extra_features = pd.DataFrame.from_dict(all_features, orient='index')
        user_features = user_features.join(extra_features)
        
        # Fill NaN values with 0
        user_features = user_features.fillna(0)
        
        # Add derived features using vectorized operations
        user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
        user_features['rating_frequency'] = user_features['rating_count'] / (user_features['time_span'] + 1)
        user_features['is_active'] = (user_features['rating_count'] > user_features['rating_count'].quantile(0.75)).astype(np.int8)
        user_features['is_diverse'] = (user_features['movie_diversity'] > user_features['movie_diversity'].quantile(0.75)).astype(np.int8)
        
        # Convert to optimal dtypes
        user_features = self.optimize_dtypes_ultra(user_features)
        
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features in ULTRA-FAST mode[/green]")
        return user_features
    
    def create_movie_features_ultra_fast(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast movie feature creation."""
        self.console.print("[cyan]Creating movie features (ULTRA-FAST mode)...[/cyan]")
        
        # Use swifter for faster pandas operations
        movie_stats = ratings_df.swifter.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique',
            'timestamp': ['min', 'max']
        })
        
        movie_stats.columns = ['rating_count', 'rating_mean', 'rating_std', 
                              'rating_min', 'rating_max', 'user_diversity',
                              'first_rating', 'last_rating']
        
        # Join with movie metadata
        movie_features = movie_stats.join(movies_df.set_index('movieId'), how='left')
        
        # Vectorized genre processing
        if 'genre_list' in movie_features.columns:
            # Create genre matrix using MultiLabelBinarizer (faster than manual)
            mlb = MultiLabelBinarizer(sparse_output=True)
            genre_matrix = mlb.fit_transform(movie_features['genre_list'].fillna(''))
            
            # Convert to DataFrame (keep sparse for memory efficiency)
            genre_df = pd.DataFrame.sparse.from_spmatrix(
                genre_matrix,
                index=movie_features.index,
                columns=[f'genre_{g}'.lower().replace(' ', '_') for g in mlb.classes_]
            )
            
            movie_features = pd.concat([movie_features, genre_df], axis=1)
            movie_features['genre_count'] = genre_matrix.sum(axis=1).A1
        
        # Fast popularity calculation
        movie_features['popularity_score'] = movie_features['rating_count'] * movie_features['rating_mean']
        movie_features['rating_velocity'] = movie_features['rating_count'] / (
            (movie_features['last_rating'] - movie_features['first_rating']).dt.total_seconds() / 86400 + 1
        )
        
        # Binary features
        movie_features['is_popular'] = (movie_features['rating_count'] > movie_features['rating_count'].quantile(0.8)).astype(np.int8)
        movie_features['is_acclaimed'] = (movie_features['rating_mean'] > 4.0).astype(np.int8)
        
        # Movie age
        current_year = pd.Timestamp.now().year
        movie_features['movie_age'] = current_year - movie_features['year'].fillna(current_year)
        
        movie_features = self.optimize_dtypes_ultra(movie_features.fillna(0))
        
        self.console.print(f"[green]✓ Created {len(movie_features.columns)} movie features in ULTRA-FAST mode[/green]")
        return movie_features
    
    def create_temporal_features_vectorized(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized temporal feature creation."""
        self.console.print("[cyan]Creating temporal features (vectorized)...[/cyan]")
        
        temp_df = ratings_df.copy()
        
        # Vectorized datetime operations
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        temp_df['day_of_week'] = temp_df['timestamp'].dt.dayofweek
        temp_df['month'] = temp_df['timestamp'].dt.month
        temp_df['year'] = temp_df['timestamp'].dt.year
        temp_df['day_of_year'] = temp_df['timestamp'].dt.dayofyear
        
        # Vectorized season calculation
        temp_df['season'] = (temp_df['month'] % 12 + 3) // 3
        
        # Weekend flag
        temp_df['is_weekend'] = (temp_df['day_of_week'] >= 5).astype(np.int8)
        
        # Time of day categories
        temp_df['time_of_day'] = pd.cut(
            temp_df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        return temp_df
    
    def create_interaction_features_parallel(self, df: pd.DataFrame, user_features: pd.DataFrame, 
                                           movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features using parallel processing."""
        self.console.print("[cyan]Creating interaction features (parallel)...[/cyan]")
        
        # Merge features efficiently
        df = df.merge(user_features[['rating_mean', 'rating_count']], 
                     left_on='userId', right_index=True, suffixes=('', '_user'))
        df = df.merge(movie_features[['rating_mean', 'rating_count']], 
                     left_on='movieId', right_index=True, suffixes=('', '_movie'))
        
        # Vectorized interaction features
        df['user_movie_rating_diff'] = df['rating_mean_user'] - df['rating_mean_movie']
        df['user_movie_count_ratio'] = df['rating_count_user'] / (df['rating_count_movie'] + 1)
        df['rating_vs_user_mean'] = df['rating'] - df['rating_mean_user']
        df['rating_vs_movie_mean'] = df['rating'] - df['rating_mean_movie']
        
        return df
    
    def create_sparse_features_ultra_fast(self, tags_df: pd.DataFrame, movies_df: pd.DataFrame,
                                         max_features: int = 200) -> Tuple[csr_matrix, np.ndarray, TfidfVectorizer]:
        """Ultra-fast sparse feature creation."""
        self.console.print("[cyan]Creating sparse TF-IDF features (ultra-fast)...[/cyan]")
        
        # Aggregate tags per movie
        movie_tags = tags_df.groupby('movieId')['tag_clean'].apply(' '.join).reset_index()
        
        # Ensure all movies are included
        all_movies = movies_df[['movieId']].copy()
        movie_tags = all_movies.merge(movie_tags, on='movieId', how='left')
        movie_tags['tag_clean'] = movie_tags['tag_clean'].fillna('')
        
        # Use HashingVectorizer for speed (no vocabulary needed)
        from sklearn.feature_extraction.text import HashingVectorizer
        
        vectorizer = HashingVectorizer(
            n_features=max_features,
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2'
        )
        
        # Process in parallel chunks
        chunk_size = len(movie_tags) // self.n_jobs
        chunks = [movie_tags['tag_clean'].iloc[i:i+chunk_size] for i in range(0, len(movie_tags), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            sparse_chunks = list(executor.map(vectorizer.transform, chunks))
        
        # Combine chunks
        tfidf_matrix = sparse.vstack(sparse_chunks)
        
        self.console.print(f"[green]✓ Created {tfidf_matrix.shape} sparse matrix[/green]")
        return tfidf_matrix, movie_tags['movieId'].values, vectorizer
    
    def create_user_item_matrix_fast(self, ratings_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Ultra-fast user-item matrix creation."""
        self.console.print("[cyan]Creating user-item matrix (optimized)...[/cyan]")
        
        # Get unique users and movies
        users = ratings_df['userId'].unique()
        movies = ratings_df['movieId'].unique()
        
        # Create mappings
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(movies)}
        
        # Create COO matrix (fastest for construction)
        row_ind = ratings_df['userId'].map(user_to_idx).values
        col_ind = ratings_df['movieId'].map(movie_to_idx).values
        data = ratings_df['rating'].values.astype(np.float32)
        
        matrix = coo_matrix((data, (row_ind, col_ind)), 
                           shape=(len(users), len(movies)), 
                           dtype=np.float32)
        
        # Convert to CSR for efficient operations
        matrix = matrix.tocsr()
        
        self.console.print(f"[green]✓ Created {matrix.shape} sparse matrix (density: {matrix.nnz / matrix.shape[0] / matrix.shape[1]:.4f})[/green]")
        
        return matrix, user_to_idx, movie_to_idx
    
    @lru_cache(maxsize=128)
    def _cached_computation(self, key: str, *args):
        """Cache expensive computations."""
        return self.transformation_cache.get(key, None)
    
    def parallel_transform_batches(self, df: pd.DataFrame, transform_func, batch_size: int = 500000):
        """Transform data in parallel batches."""
        n_batches = (len(df) + batch_size - 1) // batch_size
        
        def process_batch(i):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            return transform_func(df.iloc[start_idx:end_idx])
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(process_batch, range(n_batches)))
        
        return pd.concat(results, ignore_index=True)
    
    def create_cold_start_features_ultra_fast(self, ratings_df: pd.DataFrame, user_features: pd.DataFrame,
                                             movie_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ultra-fast cold start feature creation."""
        self.console.print("[cyan]Creating cold start features (ultra-fast)...[/cyan]")
        
        # User cold start features
        user_cold_features = pd.DataFrame(index=user_features.index)
        
        # Vectorized operations
        cold_threshold = user_features['rating_count'].quantile(0.1)
        user_cold_features['is_cold_start'] = (user_features['rating_count'] < cold_threshold).astype(np.int8)
        user_cold_features['rating_confidence'] = 1 - np.exp(-user_features['rating_count'] / 10)
        
        # Fast clustering using MiniBatchKMeans
        n_clusters = min(50, len(user_features) // 1000)
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 3:
            # Use only most important features for clustering
            cluster_features = user_features[['rating_count', 'rating_mean', 'rating_std']].fillna(0)
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=10000,
                random_state=42,
                n_init=3,
                max_iter=50
            )
            user_cold_features['user_cluster'] = kmeans.fit_predict(cluster_features)
            
            # Fast cluster statistics
            cluster_means = ratings_df.groupby(
                ratings_df['userId'].map(dict(zip(user_features.index, user_cold_features['user_cluster'])))
            )['rating'].mean()
            user_cold_features['cluster_avg_rating'] = user_cold_features['user_cluster'].map(cluster_means).fillna(3.5)
        
        # Movie cold start features
        movie_cold_features = pd.DataFrame(index=movie_features.index)
        
        movie_cold_threshold = movie_features['rating_count'].quantile(0.1)
        movie_cold_features['is_cold_start'] = (movie_features['rating_count'] < movie_cold_threshold).astype(np.int8)
        movie_cold_features['rating_confidence'] = 1 - np.exp(-movie_features['rating_count'] / 10)
        
        # Fast movie clustering
        if len(movie_features) > 100:
            genre_cols = [col for col in movie_features.columns if col.startswith('genre_')]
            if len(genre_cols) > 5:
                # Use binary genre features for clustering
                movie_cluster_features = movie_features[genre_cols[:20]].fillna(0)
                
                movie_kmeans = MiniBatchKMeans(
                    n_clusters=min(100, len(movie_features) // 500),
                    batch_size=5000,
                    random_state=42,
                    n_init=3,
                    max_iter=50
                )
                movie_cold_features['movie_cluster'] = movie_kmeans.fit_predict(movie_cluster_features)
        
        # Global average
        global_avg = ratings_df['rating'].mean()
        user_cold_features['global_avg_rating'] = global_avg
        movie_cold_features['global_avg_rating'] = global_avg
        
        return user_cold_features, movie_cold_features
    
    def create_enhanced_genre_features_ultra_fast(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Ultra-fast genre feature creation using vectorized operations."""
        self.console.print("[cyan]Creating enhanced genre features (ultra-fast)...[/cyan]")
        
        # Binary genre encoding using sparse matrix
        mlb = MultiLabelBinarizer(sparse_output=True)
        genre_matrix = mlb.fit_transform(movies_df['genre_list'].fillna(''))
        
        # Convert to sparse DataFrame
        genre_df = pd.DataFrame.sparse.from_spmatrix(
            genre_matrix,
            index=movies_df['movieId'],
            columns=[f'genre_{g}'.lower().replace(' ', '_') for g in mlb.classes_]
        )
        
        # Basic genre statistics
        genre_stats = pd.DataFrame(index=movies_df['movieId'])
        genre_stats['genre_count'] = genre_matrix.sum(axis=1).A1
        genre_stats['is_single_genre'] = (genre_stats['genre_count'] == 1).astype(np.int8)
        genre_stats['is_multi_genre'] = (genre_stats['genre_count'] > 1).astype(np.int8)
        
        # Fast genre performance using groupby
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).fillna(0)
        movie_stats.columns = ['avg_rating', 'num_ratings', 'rating_std']
        
        # Combine features
        genre_features = pd.concat([genre_df, genre_stats], axis=1)
        
        return genre_features
    
    def create_all_features_pipeline(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                                   tags_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Complete feature engineering pipeline optimized for speed."""
        self.console.print("\n[bold green]Starting ULTRA-FAST Feature Engineering Pipeline[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Feature Engineering...", total=8)
            
            # 1. Optimize dtypes
            progress.update(task, description="[cyan]Optimizing data types...")
            ratings_df = self.optimize_dtypes_ultra(ratings_df)
            movies_df = self.optimize_dtypes_ultra(movies_df)
            progress.advance(task)
            
            # 2. Create temporal features
            progress.update(task, description="[cyan]Creating temporal features...")
            ratings_df = self.create_temporal_features_vectorized(ratings_df)
            progress.advance(task)
            
            # 3. Create user features (parallel)
            progress.update(task, description="[cyan]Creating user features...")
            user_features = self.create_user_features_ultra_fast(ratings_df)
            progress.advance(task)
            
            # 4. Create movie features (parallel)
            progress.update(task, description="[cyan]Creating movie features...")
            movie_features = self.create_movie_features_ultra_fast(ratings_df, movies_df)
            progress.advance(task)
            
            # 5. Create genre features
            progress.update(task, description="[cyan]Creating genre features...")
            genre_features = self.create_enhanced_genre_features_ultra_fast(movies_df, ratings_df)
            movie_features = pd.concat([movie_features, genre_features], axis=1)
            progress.advance(task)
            
            # 6. Create cold start features
            progress.update(task, description="[cyan]Creating cold start features...")
            user_cold, movie_cold = self.create_cold_start_features_ultra_fast(
                ratings_df, user_features, movie_features
            )
            user_features = pd.concat([user_features, user_cold], axis=1)
            movie_features = pd.concat([movie_features, movie_cold], axis=1)
            progress.advance(task)
            
            # 7. Create interaction features
            progress.update(task, description="[cyan]Creating interaction features...")
            ratings_enhanced = self.create_interaction_features_parallel(
                ratings_df, user_features, movie_features
            )
            progress.advance(task)
            
            # 8. Create sparse features if tags available
            sparse_features = None
            if tags_df is not None and len(tags_df) > 0:
                progress.update(task, description="[cyan]Creating sparse features...")
                sparse_matrix, movie_ids, vectorizer = self.create_sparse_features_ultra_fast(
                    tags_df, movies_df
                )
                sparse_features = {
                    'matrix': sparse_matrix,
                    'movie_ids': movie_ids,
                    'vectorizer': vectorizer
                }
            progress.advance(task)
        
        return {
            'ratings_enhanced': ratings_enhanced,
            'user_features': user_features,
            'movie_features': movie_features,
            'sparse_features': sparse_features
        }