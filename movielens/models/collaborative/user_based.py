"""
User-Based Collaborative Filtering for MovieLens Dataset
Location: movielens/models/collaborative/user_based.py

This module implements user-based collaborative filtering with:
- Multiple similarity metrics (cosine, pearson, jaccard)
- Neighborhood selection strategies
- Prediction methods with bias correction
- Integration with clustering results for improved performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine, correlation
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
import multiprocessing as mp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
import warnings
import pickle
import gzip
from pathlib import Path
import time
from ...config import PROCESSED_DATA_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
console = Console()


class UserBasedCF:
    """
    User-based collaborative filtering with enhanced features.
    """
    
    def __init__(self, 
                 k_neighbors: int = 50,
                 min_common_items: int = 5,
                 similarity_metric: str = 'cosine',
                 use_bias_correction: bool = True,
                 n_jobs: int = -1):
        """
        Initialize user-based collaborative filtering.
        
        Args:
            k_neighbors: Number of similar users to consider
            min_common_items: Minimum items in common for similarity calculation
            similarity_metric: 'cosine', 'pearson', or 'jaccard'
            use_bias_correction: Whether to use bias correction in predictions
            n_jobs: Number of parallel jobs
        """
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.similarity_metric = similarity_metric
        self.use_bias_correction = use_bias_correction
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        self.user_item_matrix = None
        self.user_similarities = None
        self.user_means = None
        self.user_mapping = None
        self.movie_mapping = None
        self.reverse_user_mapping = None
        self.reverse_movie_mapping = None
        
        self.console = Console()
    
    def fit(self, ratings_df: pd.DataFrame, 
            user_clusters: Optional[pd.Series] = None):
        """
        Fit the user-based collaborative filtering model.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            user_clusters: Optional user cluster assignments for improved performance
        """
        self.console.print(Panel.fit(
            "[bold cyan]Training User-Based Collaborative Filtering[/bold cyan]\n"
            f"Neighbors: {self.k_neighbors}, Metric: {self.similarity_metric}",
            border_style="cyan"
        ))
        
        # Create user-item matrix
        self._create_user_item_matrix(ratings_df)
        
        # Calculate user similarities
        if user_clusters is not None:
            self.console.print("[cyan]Using cluster information for similarity computation...[/cyan]")
            self._calculate_similarities_with_clusters(user_clusters)
        else:
            self._calculate_similarities()
        
        # Calculate user biases
        if self.use_bias_correction:
            self._calculate_biases()
        
        self.console.print("[green]✓ User-based CF model trained successfully[/green]")
    
    def _create_user_item_matrix(self, ratings_df: pd.DataFrame):
        """Create sparse user-item rating matrix."""
        self.console.print("[cyan]Creating user-item matrix...[/cyan]")
        
        # Create mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_movie_mapping = {idx: movie for movie, idx in self.movie_mapping.items()}
        
        # Create sparse matrix
        row_indices = ratings_df['userId'].map(self.user_mapping)
        col_indices = ratings_df['movieId'].map(self.movie_mapping)
        ratings = ratings_df['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_movies))
        )
        
        self.console.print(f"[green]✓ Created {self.user_item_matrix.shape} user-item matrix[/green]")
    
    def _calculate_similarities(self):
        """Calculate user similarities using the specified metric."""
        self.console.print(f"[cyan]Calculating user similarities ({self.similarity_metric})...[/cyan]")
        
        start_time = time.time()
        
        if self.similarity_metric == 'cosine':
            # Use sklearn's optimized cosine similarity
            self.user_similarities = cosine_similarity(self.user_item_matrix, dense_output=False)
        
        elif self.similarity_metric == 'pearson':
            # Pearson correlation (centered cosine similarity)
            user_means = self.user_item_matrix.mean(axis=1).A1
            centered_matrix = self.user_item_matrix.copy()
            
            # Center the matrix
            for i in range(centered_matrix.shape[0]):
                row = centered_matrix.getrow(i)
                if row.nnz > 0:
                    centered_matrix.data[row.indptr[0]:row.indptr[1]] -= user_means[i]
            
            self.user_similarities = cosine_similarity(centered_matrix, dense_output=False)
        
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity for implicit feedback
            binary_matrix = (self.user_item_matrix > 0).astype(float)
            self.user_similarities = self._calculate_jaccard_parallel(binary_matrix)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (no self-similarity)
        self.user_similarities.setdiag(0)
        
        elapsed_time = time.time() - start_time
        self.console.print(f"[green]✓ Similarities calculated in {elapsed_time:.2f}s[/green]")
    
    def _calculate_jaccard_parallel(self, binary_matrix: csr_matrix) -> csr_matrix:
        """Calculate Jaccard similarity in parallel."""
        n_users = binary_matrix.shape[0]
        
        def compute_jaccard_row(i):
            row_i = binary_matrix.getrow(i)
            similarities = []
            indices = []
            
            for j in range(n_users):
                if i != j:
                    row_j = binary_matrix.getrow(j)
                    intersection = (row_i.multiply(row_j)).sum()
                    union = ((row_i + row_j) > 0).sum()
                    
                    if union > 0:
                        similarity = intersection / union
                        if similarity > 0:
                            similarities.append(similarity)
                            indices.append(j)
            
            return i, indices, similarities
        
        # Parallel computation
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_jaccard_row)(i) for i in range(n_users)
        )
        
        # Build sparse similarity matrix
        similarity_matrix = lil_matrix((n_users, n_users))
        for i, indices, similarities in results:
            similarity_matrix[i, indices] = similarities
        
        return similarity_matrix.tocsr()
    
    def _calculate_similarities_with_clusters(self, user_clusters: pd.Series):
        """Calculate similarities considering user clusters."""
        # First calculate base similarities
        self._calculate_similarities()
        
        # Boost similarities within same cluster
        cluster_boost = 1.2
        
        for user_id, cluster in user_clusters.items():
            if user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                
                # Find other users in same cluster
                same_cluster_users = user_clusters[user_clusters == cluster].index
                
                for other_user in same_cluster_users:
                    if other_user != user_id and other_user in self.user_mapping:
                        other_idx = self.user_mapping[other_user]
                        
                        # Boost similarity
                        current_sim = self.user_similarities[user_idx, other_idx]
                        self.user_similarities[user_idx, other_idx] = min(
                            current_sim * cluster_boost, 1.0
                        )
    
    def _calculate_biases(self):
        """Calculate user bias (mean rating deviation)."""
        self.console.print("[cyan]Calculating user biases...[/cyan]")
        
        # Calculate global mean
        self.global_mean = self.user_item_matrix.data.mean()
        
        # Calculate user means
        self.user_means = np.zeros(self.user_item_matrix.shape[0])
        for i in range(self.user_item_matrix.shape[0]):
            row = self.user_item_matrix.getrow(i)
            if row.nnz > 0:
                self.user_means[i] = row.data.mean()
            else:
                self.user_means[i] = self.global_mean
        
        # Calculate user biases
        self.user_biases = self.user_means - self.global_mean
    
    def predict(self, user_id: int, movie_id: int) -> Optional[float]:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if prediction not possible
        """
        if user_id not in self.user_mapping or movie_id not in self.movie_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        movie_idx = self.movie_mapping[movie_id]
        
        # Get k most similar users who have rated this movie
        similarities = self.user_similarities[user_idx].toarray().flatten()
        
        # Find users who rated this movie
        movie_raters = self.user_item_matrix[:, movie_idx].nonzero()[0]
        
        if len(movie_raters) == 0:
            return self.user_means[user_idx] if hasattr(self, 'user_means') else None
        
        # Get similarities for users who rated the movie
        rater_similarities = similarities[movie_raters]
        
        # Select top k neighbors
        if len(movie_raters) > self.k_neighbors:
            top_k_indices = np.argpartition(rater_similarities, -self.k_neighbors)[-self.k_neighbors:]
            neighbor_indices = movie_raters[top_k_indices]
            neighbor_similarities = rater_similarities[top_k_indices]
        else:
            neighbor_indices = movie_raters
            neighbor_similarities = rater_similarities
        
        # Filter by minimum similarity threshold
        valid_neighbors = neighbor_similarities > 0
        neighbor_indices = neighbor_indices[valid_neighbors]
        neighbor_similarities = neighbor_similarities[valid_neighbors]
        
        if len(neighbor_indices) == 0:
            return self.user_means[user_idx] if hasattr(self, 'user_means') else None
        
        # Get neighbor ratings
        neighbor_ratings = self.user_item_matrix[neighbor_indices, movie_idx].toarray().flatten()
        
        if self.use_bias_correction:
            # Adjust for user biases
            adjusted_ratings = neighbor_ratings - self.user_biases[neighbor_indices]
            prediction = self.user_means[user_idx] + \
                        np.sum(neighbor_similarities * adjusted_ratings) / np.sum(neighbor_similarities)
        else:
            # Weighted average
            prediction = np.sum(neighbor_similarities * neighbor_ratings) / np.sum(neighbor_similarities)
        
        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)
    
    def predict_batch(self, user_movie_pairs: List[Tuple[int, int]]) -> List[Optional[float]]:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs: List of (user_id, movie_id) tuples
            
        Returns:
            List of predicted ratings
        """
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict)(user_id, movie_id) 
            for user_id, movie_id in user_movie_pairs
        )
        
        return predictions
    
    def recommend_movies(self, user_id: int, n_recommendations: int = 10,
                        exclude_watched: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend movies for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_watched: Whether to exclude already watched movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.getrow(user_idx)
        rated_movies = set(user_ratings.indices)
        
        if exclude_watched:
            candidate_movies = [
                movie_idx for movie_idx in range(self.user_item_matrix.shape[1])
                if movie_idx not in rated_movies
            ]
        else:
            candidate_movies = list(range(self.user_item_matrix.shape[1]))
        
        # Predict ratings for candidate movies
        predictions = []
        for movie_idx in candidate_movies:
            movie_id = self.reverse_movie_mapping[movie_idx]
            rating = self.predict(user_id, movie_id)
            if rating is not None:
                predictions.append((movie_id, rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def evaluate(self, test_ratings: pd.DataFrame, 
                verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on test ratings.
        
        Args:
            test_ratings: DataFrame with test ratings
            verbose: Whether to display progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            self.console.print("[cyan]Evaluating user-based CF model...[/cyan]")
        
        # Filter test ratings to users/movies in training
        valid_mask = (
            test_ratings['userId'].isin(self.user_mapping.keys()) &
            test_ratings['movieId'].isin(self.movie_mapping.keys())
        )
        valid_test = test_ratings[valid_mask]
        
        if len(valid_test) == 0:
            self.console.print("[red]No valid test samples found![/red]")
            return {}
        
        # Make predictions
        user_movie_pairs = list(zip(valid_test['userId'], valid_test['movieId']))
        
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Predicting {len(user_movie_pairs):,} ratings...", 
                    total=1
                )
                predictions = self.predict_batch(user_movie_pairs)
                progress.advance(task)
        else:
            predictions = self.predict_batch(user_movie_pairs)
        
        # Filter out None predictions
        valid_predictions = [
            (true, pred) for true, pred in zip(valid_test['rating'], predictions)
            if pred is not None
        ]
        
        if not valid_predictions:
            self.console.print("[red]No valid predictions made![/red]")
            return {}
        
        true_ratings, pred_ratings = zip(*valid_predictions)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)
        
        # Coverage
        coverage = len(valid_predictions) / len(valid_test)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'n_predictions': len(valid_predictions)
        }
        
        if verbose:
            self._display_evaluation_results(metrics)
        
        return metrics
    
    def _display_evaluation_results(self, metrics: Dict[str, float]):
        """Display evaluation results."""
        table = Table(title="User-Based CF Evaluation Results", box="rounded")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("RMSE", f"{metrics['rmse']:.4f}")
        table.add_row("MAE", f"{metrics['mae']:.4f}")
        table.add_row("Coverage", f"{metrics['coverage']:.2%}")
        table.add_row("Valid Predictions", f"{metrics['n_predictions']:,}")
        
        self.console.print(table)
    
    def save_model(self, filepath: Optional[Path] = None):
        """Save the trained model."""
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "user_based_cf_model.pkl.gz"
        
        model_data = {
            'k_neighbors': self.k_neighbors,
            'min_common_items': self.min_common_items,
            'similarity_metric': self.similarity_metric,
            'use_bias_correction': self.use_bias_correction,
            'user_item_matrix': self.user_item_matrix,
            'user_similarities': self.user_similarities,
            'user_means': self.user_means,
            'user_biases': getattr(self, 'user_biases', None),
            'global_mean': getattr(self, 'global_mean', None),
            'user_mapping': self.user_mapping,
            'movie_mapping': self.movie_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_movie_mapping': self.reverse_movie_mapping
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
        
        self.console.print(f"[green]✓ Model saved to {filepath}[/green]")
    
    def load_model(self, filepath: Optional[Path] = None):
        """Load a saved model."""
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "user_based_cf_model.pkl.gz"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with gzip.open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore attributes
        for key, value in model_data.items():
            setattr(self, key, value)
        
        self.console.print(f"[green]✓ Model loaded from {filepath}[/green]")


def run_user_based_cf_pipeline(ratings_df: pd.DataFrame,
                              test_size: float = 0.2,
                              k_neighbors: List[int] = [20, 50, 100],
                              similarity_metrics: List[str] = ['cosine', 'pearson'],
                              user_clusters: Optional[pd.Series] = None) -> Dict:
    """
    Run complete user-based CF pipeline with hyperparameter tuning.
    
    Args:
        ratings_df: DataFrame with ratings
        test_size: Fraction of data for testing
        k_neighbors: List of k values to test
        similarity_metrics: List of similarity metrics to test
        user_clusters: Optional user cluster assignments
        
    Returns:
        Dictionary with results and best model
    """
    console.print(Panel.fit(
        "[bold cyan]User-Based Collaborative Filtering Pipeline[/bold cyan]\n"
        "Training and evaluating with multiple configurations",
        border_style="cyan"
    ))
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=test_size, random_state=42
    )
    
    console.print(f"[green]✓ Data split: {len(train_ratings):,} train, {len(test_ratings):,} test[/green]")
    
    # Test different configurations
    results = {}
    best_model = None
    best_rmse = float('inf')
    
    for metric in similarity_metrics:
        for k in k_neighbors:
            console.print(f"\n[cyan]Testing {metric} similarity with k={k}...[/cyan]")
            
            # Create and train model
            model = UserBasedCF(
                k_neighbors=k,
                similarity_metric=metric,
                use_bias_correction=True
            )
            model.fit(train_ratings, user_clusters)
            
            # Evaluate
            metrics = model.evaluate(test_ratings)
            
            # Store results
            config_name = f"{metric}_k{k}"
            results[config_name] = {
                'model': model,
                'metrics': metrics,
                'config': {
                    'similarity_metric': metric,
                    'k_neighbors': k
                }
            }
            
            # Check if best
            if metrics.get('rmse', float('inf')) < best_rmse:
                best_rmse = metrics['rmse']
                best_model = model
                best_config = config_name
    
    # Display comparison
    comparison_table = Table(title="User-Based CF Configuration Comparison", box="rounded")
    comparison_table.add_column("Configuration", style="cyan")
    comparison_table.add_column("RMSE", justify="right")
    comparison_table.add_column("MAE", justify="right")
    comparison_table.add_column("Coverage", justify="right")
    
    for config_name, result in results.items():
        metrics = result['metrics']
        comparison_table.add_row(
            config_name,
            f"{metrics.get('rmse', 'N/A'):.4f}" if 'rmse' in metrics else "N/A",
            f"{metrics.get('mae', 'N/A'):.4f}" if 'mae' in metrics else "N/A",
            f"{metrics.get('coverage', 0):.2%}"
        )
    
    console.print(comparison_table)
    console.print(f"\n[green]✓ Best configuration: {best_config} (RMSE: {best_rmse:.4f})[/green]")
    
    # Save best model
    if best_model:
        best_model.save_model()
    
    return {
        'results': results,
        'best_model': best_model,
        'best_config': best_config
    }


if __name__ == "__main__":
    # Example usage
    console.print("[yellow]This module should be imported and used through the main pipeline[/yellow]")
    console.print("Example usage:")
    console.print("from movielens.models.collaborative.user_based import run_user_based_cf_pipeline")
    console.print("results = run_user_based_cf_pipeline(ratings_df)")
