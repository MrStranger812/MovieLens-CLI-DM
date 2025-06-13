"""
Item-Based Collaborative Filtering for MovieLens Dataset
Location: movielens/models/collaborative/item_based.py

This module implements item-based collaborative filtering with:
- Efficient similarity computation for large item catalogs
- Multiple similarity metrics
- Baseline predictors integration
- Matrix factorization enhancements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
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


class ItemBasedCF:
    """
    Item-based collaborative filtering with optimizations for large catalogs.
    """
    
    def __init__(self,
                 k_neighbors: int = 30,
                 min_common_users: int = 10,
                 similarity_metric: str = 'cosine',
                 use_baseline: bool = True,
                 shrinkage: float = 100,
                 n_jobs: int = -1):
        """
        Initialize item-based collaborative filtering.
        
        Args:
            k_neighbors: Number of similar items to consider
            min_common_users: Minimum users in common for similarity
            similarity_metric: 'cosine', 'adjusted_cosine', or 'correlation'
            use_baseline: Whether to use baseline predictors
            shrinkage: Shrinkage parameter for similarity computation
            n_jobs: Number of parallel jobs
        """
        self.k_neighbors = k_neighbors
        self.min_common_users = min_common_users
        self.similarity_metric = similarity_metric
        self.use_baseline = use_baseline
        self.shrinkage = shrinkage
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        self.item_user_matrix = None
        self.item_similarities = None
        self.item_means = None
        self.user_means = None
        self.global_mean = None
        self.user_mapping = None
        self.movie_mapping = None
        self.reverse_user_mapping = None
        self.reverse_movie_mapping = None
        
        self.console = Console()
    
    def fit(self, ratings_df: pd.DataFrame,
            movie_features: Optional[pd.DataFrame] = None):
        """
        Fit the item-based collaborative filtering model.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            movie_features: Optional movie features for hybrid similarity
        """
        self.console.print(Panel.fit(
            "[bold cyan]Training Item-Based Collaborative Filtering[/bold cyan]\n"
            f"Neighbors: {self.k_neighbors}, Metric: {self.similarity_metric}",
            border_style="cyan"
        ))
        
        # Create item-user matrix (transpose of user-item)
        self._create_item_user_matrix(ratings_df)
        
        # Calculate baselines
        if self.use_baseline:
            self._calculate_baselines()
        
        # Calculate item similarities
        if movie_features is not None:
            self.console.print("[cyan]Using hybrid similarity with content features...[/cyan]")
            self._calculate_hybrid_similarities(movie_features)
        else:
            self._calculate_similarities()
        
        self.console.print("[green]✓ Item-based CF model trained successfully[/green]")
    
    def _create_item_user_matrix(self, ratings_df: pd.DataFrame):
        """Create sparse item-user rating matrix."""
        self.console.print("[cyan]Creating item-user matrix...[/cyan]")
        
        # Create mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_movie_mapping = {idx: movie for movie, idx in self.movie_mapping.items()}
        
        # Create sparse matrix (items x users)
        row_indices = ratings_df['movieId'].map(self.movie_mapping)
        col_indices = ratings_df['userId'].map(self.user_mapping)
        ratings = ratings_df['rating'].values
        
        self.item_user_matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(len(unique_movies), len(unique_users))
        )
        
        self.console.print(f"[green]✓ Created {self.item_user_matrix.shape} item-user matrix[/green]")
    
    def _calculate_baselines(self):
        """Calculate baseline predictors (global mean, user/item biases)."""
        self.console.print("[cyan]Calculating baseline predictors...[/cyan]")
        
        # Global mean
        self.global_mean = self.item_user_matrix.data.mean()
        
        # Item means
        self.item_means = np.zeros(self.item_user_matrix.shape[0])
        for i in range(self.item_user_matrix.shape[0]):
            row = self.item_user_matrix.getrow(i)
            if row.nnz > 0:
                self.item_means[i] = row.data.mean()
            else:
                self.item_means[i] = self.global_mean
        
        # User means
        self.user_means = np.zeros(self.item_user_matrix.shape[1])
        for j in range(self.item_user_matrix.shape[1]):
            col = self.item_user_matrix.getcol(j)
            if col.nnz > 0:
                self.user_means[j] = col.data.mean()
            else:
                self.user_means[j] = self.global_mean
        
        # Calculate biases
        self.item_biases = self.item_means - self.global_mean
        self.user_biases = self.user_means - self.global_mean
    
    def _calculate_similarities(self):
        """Calculate item similarities using the specified metric."""
        self.console.print(f"[cyan]Calculating item similarities ({self.similarity_metric})...[/cyan]")
        
        start_time = time.time()
        
        if self.similarity_metric == 'cosine':
            # Standard cosine similarity
            self.item_similarities = cosine_similarity(self.item_user_matrix, dense_output=False)
        
        elif self.similarity_metric == 'adjusted_cosine':
            # Adjust for user mean ratings
            adjusted_matrix = self.item_user_matrix.copy().astype(float)
            
            # Subtract user means from each rating
            for j in range(adjusted_matrix.shape[1]):
                col = adjusted_matrix.getcol(j)
                if col.nnz > 0:
                    adjusted_matrix.data[col.indices] -= self.user_means[j]
            
            self.item_similarities = cosine_similarity(adjusted_matrix, dense_output=False)
        
        elif self.similarity_metric == 'correlation':
            # Pearson correlation (item-centered cosine)
            centered_matrix = self.item_user_matrix.copy().astype(float)
            
            # Center by item means
            for i in range(centered_matrix.shape[0]):
                row = centered_matrix.getrow(i)
                if row.nnz > 0:
                    centered_matrix.data[row.indices] -= self.item_means[i]
            
            self.item_similarities = cosine_similarity(centered_matrix, dense_output=False)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Apply shrinkage
        if self.shrinkage > 0:
            self._apply_shrinkage()
        
        # Set diagonal to 0
        self.item_similarities.setdiag(0)
        
        elapsed_time = time.time() - start_time
        self.console.print(f"[green]✓ Similarities calculated in {elapsed_time:.2f}s[/green]")
    
    def _apply_shrinkage(self):
        """Apply shrinkage to similarities based on number of common users."""
        self.console.print("[cyan]Applying similarity shrinkage...[/cyan]")
        
        # Calculate number of common users for each item pair
        binary_matrix = (self.item_user_matrix > 0).astype(float)
        common_users = binary_matrix @ binary_matrix.T
        
        # Apply shrinkage formula
        shrinkage_factor = common_users / (common_users + self.shrinkage)
        self.item_similarities = self.item_similarities.multiply(shrinkage_factor)
    
    def _calculate_hybrid_similarities(self, movie_features: pd.DataFrame):
        """Calculate hybrid similarities using ratings and content features."""
        # First calculate collaborative similarities
        self._calculate_similarities()
        
        # Calculate content-based similarities
        self.console.print("[cyan]Calculating content-based similarities...[/cyan]")
        
        # Align movie features with our movie mapping
        feature_matrix = []
        for movie_id in sorted(self.movie_mapping.keys()):
            if movie_id in movie_features.index:
                feature_matrix.append(movie_features.loc[movie_id].values)
            else:
                feature_matrix.append(np.zeros(movie_features.shape[1]))
        
        feature_matrix = np.array(feature_matrix)
        content_similarities = cosine_similarity(feature_matrix, dense_output=False)
        
        # Combine similarities (weighted average)
        cf_weight = 0.8  # Collaborative filtering weight
        cb_weight = 0.2  # Content-based weight
        
        self.item_similarities = (cf_weight * self.item_similarities + 
                                 cb_weight * content_similarities)
    
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
        
        # Get user's rated items
        user_ratings = self.item_user_matrix[:, user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            # Return baseline if available
            if self.use_baseline:
                return self.global_mean + self.user_biases[user_idx] + self.item_biases[movie_idx]
            return None
        
        # Get similarities to rated items
        similarities = self.item_similarities[movie_idx, rated_items].toarray().flatten()
        
        # Select top k neighbors
        if len(rated_items) > self.k_neighbors:
            top_k_indices = np.argpartition(similarities, -self.k_neighbors)[-self.k_neighbors:]
            neighbor_indices = rated_items[top_k_indices]
            neighbor_similarities = similarities[top_k_indices]
        else:
            neighbor_indices = rated_items
            neighbor_similarities = similarities
        
        # Filter by minimum similarity
        valid_neighbors = neighbor_similarities > 0
        neighbor_indices = neighbor_indices[valid_neighbors]
        neighbor_similarities = neighbor_similarities[valid_neighbors]
        
        if len(neighbor_indices) == 0:
            if self.use_baseline:
                return self.global_mean + self.user_biases[user_idx] + self.item_biases[movie_idx]
            return None
        
        # Get neighbor ratings
        neighbor_ratings = user_ratings[neighbor_indices]
        
        if self.use_baseline:
            # Baseline predictor
            baseline = self.global_mean + self.user_biases[user_idx] + self.item_biases[movie_idx]
            
            # Deviations from baseline
            neighbor_baselines = (self.global_mean + self.user_biases[user_idx] + 
                                self.item_biases[neighbor_indices])
            deviations = neighbor_ratings - neighbor_baselines
            
            # Weighted average of deviations
            prediction = baseline + np.sum(neighbor_similarities * deviations) / np.sum(neighbor_similarities)
        else:
            # Simple weighted average
            prediction = np.sum(neighbor_similarities * neighbor_ratings) / np.sum(neighbor_similarities)
        
        return np.clip(prediction, 0.5, 5.0)
    
    def predict_batch(self, user_movie_pairs: List[Tuple[int, int]]) -> List[Optional[float]]:
        """Predict ratings for multiple user-movie pairs."""
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
        
        # Get user's ratings
        user_ratings = self.item_user_matrix[:, user_idx].toarray().flatten()
        rated_items = set(np.where(user_ratings > 0)[0])
        
        if exclude_watched:
            candidate_items = [
                item_idx for item_idx in range(self.item_user_matrix.shape[0])
                if item_idx not in rated_items
            ]
        else:
            candidate_items = list(range(self.item_user_matrix.shape[0]))
        
        # For efficiency, only consider items similar to user's highly rated items
        highly_rated = [idx for idx in rated_items if user_ratings[idx] >= 4.0]
        
        if highly_rated:
            # Get items similar to highly rated ones
            similarity_scores = {}
            for item_idx in candidate_items:
                max_sim = 0
                for rated_idx in highly_rated:
                    sim = self.item_similarities[item_idx, rated_idx]
                    if sim > max_sim:
                        max_sim = sim
                if max_sim > 0:
                    similarity_scores[item_idx] = max_sim
            
            # Sort by similarity and take top candidates
            top_candidates = sorted(similarity_scores.items(), 
                                  key=lambda x: x[1], reverse=True)[:n_recommendations * 2]
            candidate_items = [idx for idx, _ in top_candidates]
        
        # Predict ratings for candidates
        predictions = []
        for item_idx in candidate_items[:n_recommendations * 2]:  # Get extra to ensure we have enough
            movie_id = self.reverse_movie_mapping[item_idx]
            rating = self.predict(user_id, movie_id)
            if rating is not None:
                predictions.append((movie_id, rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def evaluate(self, test_ratings: pd.DataFrame, verbose: bool = True) -> Dict[str, float]:
        """Evaluate the model on test ratings."""
        if verbose:
            self.console.print("[cyan]Evaluating item-based CF model...[/cyan]")
        
        # Filter test ratings
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
        table = Table(title="Item-Based CF Evaluation Results", box="rounded")
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
            filepath = PROCESSED_DATA_DIR / "item_based_cf_model.pkl.gz"
        
        model_data = {
            'k_neighbors': self.k_neighbors,
            'min_common_users': self.min_common_users,
            'similarity_metric': self.similarity_metric,
            'use_baseline': self.use_baseline,
            'shrinkage': self.shrinkage,
            'item_user_matrix': self.item_user_matrix,
            'item_similarities': self.item_similarities,
            'item_means': self.item_means,
            'user_means': self.user_means,
            'global_mean': self.global_mean,
            'item_biases': getattr(self, 'item_biases', None),
            'user_biases': getattr(self, 'user_biases', None),
            'user_mapping': self.user_mapping,
            'movie_mapping': self.movie_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_movie_mapping': self.reverse_movie_mapping
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
        
        self.console.print(f"[green]✓ Model saved to {filepath}[/green]")


def run_collaborative_filtering_comparison(ratings_df: pd.DataFrame,
                                         movie_features: Optional[pd.DataFrame] = None,
                                         test_size: float = 0.2) -> Dict:
    """
    Compare user-based and item-based collaborative filtering.
    
    Args:
        ratings_df: DataFrame with ratings
        movie_features: Optional movie features for hybrid similarity
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary with comparison results
    """
    console.print(Panel.fit(
        "[bold cyan]Collaborative Filtering Comparison[/bold cyan]\n"
        "User-Based vs Item-Based Methods",
        border_style="cyan"
    ))
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=test_size, random_state=42
    )
    
    results = {}
    
    # User-based CF
    console.print("\n[bold yellow]Training User-Based CF[/bold yellow]")
    user_cf = UserBasedCF(k_neighbors=50, similarity_metric='cosine')
    user_cf.fit(train_ratings)
    user_metrics = user_cf.evaluate(test_ratings)
    results['user_based'] = {
        'model': user_cf,
        'metrics': user_metrics
    }
    
    # Item-based CF
    console.print("\n[bold yellow]Training Item-Based CF[/bold yellow]")
    item_cf = ItemBasedCF(k_neighbors=30, similarity_metric='adjusted_cosine')
    item_cf.fit(train_ratings, movie_features)
    item_metrics = item_cf.evaluate(test_ratings)
    results['item_based'] = {
        'model': item_cf,
        'metrics': item_metrics
    }
    
    # Display comparison
    comparison_table = Table(title="CF Methods Comparison", box="rounded")
    comparison_table.add_column("Method", style="cyan")
    comparison_table.add_column("RMSE", justify="right")
    comparison_table.add_column("MAE", justify="right")
    comparison_table.add_column("Coverage", justify="right")
    
    for method, result in results.items():
        metrics = result['metrics']
        comparison_table.add_row(
            method.replace('_', ' ').title(),
            f"{metrics['rmse']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics['coverage']:.2%}"
        )
    
    console.print(comparison_table)
    
    # Save models
    user_cf.save_model()
    item_cf.save_model()
    
    return results


if __name__ == "__main__":
    console.print("[yellow]This module should be imported and used through the main pipeline[/yellow]")
    console.print("Example usage:")
    console.print("from movielens.models.collaborative.item_based import run_collaborative_filtering_comparison")
    console.print("results = run_collaborative_filtering_comparison(ratings_df, movie_features)")