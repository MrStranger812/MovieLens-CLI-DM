"""
Clustering Models for MovieLens Dataset
Location: movielens/models/clustering.py

This module implements:
- User Segmentation based on rating patterns
- Movie Clustering based on genres and ratings
- Hierarchical/Agglomerative Clustering for both users and movies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
import warnings
from pathlib import Path
import pickle
import gzip
from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
console = Console()


class UserSegmentation:
    """
    User Segmentation based on rating behaviors and preferences.
    Implements both K-means and Agglomerative clustering.
    """
    
    def __init__(self, n_clusters: int = 8, method: str = 'kmeans', random_state: int = 42):
        """
        Initialize user segmentation.
        
        Args:
            n_clusters: Number of clusters
            method: 'kmeans' or 'agglomerative'
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.console = Console()
        
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.user_features = None
        self.user_clusters = None
        self.cluster_profiles = None
        
    def prepare_user_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare user features for clustering.
        
        Args:
            ratings_df: DataFrame with user ratings
            
        Returns:
            DataFrame with user features
        """
        self.console.print("[cyan]Preparing user features for segmentation...[/cyan]")
        
        # Create user-movie rating matrix
        user_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Basic user statistics
        user_stats = ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique',
            'timestamp': ['min', 'max']
        })
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        # Add derived features
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        user_stats['activity_span_days'] = (
            user_stats['timestamp_max'] - user_stats['timestamp_min']
        ).dt.total_seconds() / (24 * 3600)
        user_stats['rating_frequency'] = user_stats['rating_count'] / (user_stats['activity_span_days'] + 1)
        user_stats['rating_range'] = user_stats['rating_max'] - user_stats['rating_min']
        
        # Genre preferences (if movies data available)
        if hasattr(self, 'movies_df') and self.movies_df is not None:
            genre_prefs = self._calculate_genre_preferences(ratings_df, self.movies_df)
            user_stats = user_stats.join(genre_prefs, how='left').fillna(0)
        
        # Select features for clustering
        feature_cols = [
            'rating_count', 'rating_mean', 'rating_std', 'rating_min', 'rating_max',
            'movieId_nunique', 'activity_span_days', 'rating_frequency', 'rating_range'
        ]
        
        # Add genre columns if available
        genre_cols = [col for col in user_stats.columns if col.startswith('genre_pref_')]
        feature_cols.extend(genre_cols[:10])  # Top 10 genres
        
        self.user_features = user_stats[feature_cols].fillna(0)
        
        self.console.print(f"[green]✓ Prepared {len(self.user_features.columns)} features for {len(self.user_features)} users[/green]")
        return self.user_features
    
    def _calculate_genre_preferences(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate user preferences for each genre."""
        # Merge ratings with movie genres
        ratings_with_genres = ratings_df.merge(
            movies_df[['movieId', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        # Split genres
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        # Calculate preference score for each genre
        genre_preferences = {}
        
        for genre in sorted(all_genres)[:20]:  # Top 20 genres
            # Find movies with this genre
            mask = ratings_with_genres['genres'].str.contains(genre, na=False, regex=False)
            
            # Calculate average rating for this genre per user
            genre_ratings = ratings_with_genres[mask].groupby('userId')['rating'].mean()
            genre_preferences[f'genre_pref_{genre.lower().replace(" ", "_")}'] = genre_ratings
        
        return pd.DataFrame(genre_preferences).fillna(0)
    
    def fit_clusters(self, scale: bool = True, reduce_dims: bool = True, n_components: int = 10):
        """
        Fit clustering model to user features.
        
        Args:
            scale: Whether to scale features
            reduce_dims: Whether to apply PCA
            n_components: Number of PCA components
        """
        if self.user_features is None:
            raise ValueError("No user features available. Run prepare_user_features first.")
        
        self.console.print(f"[cyan]Fitting {self.method} clustering with {self.n_clusters} clusters...[/cyan]")
        
        # Scale features
        if scale:
            user_features_scaled = self.scaler.fit_transform(self.user_features)
        else:
            user_features_scaled = self.user_features.values
        
        # Dimensionality reduction
        if reduce_dims and user_features_scaled.shape[1] > n_components:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            user_features_scaled = self.pca.fit_transform(user_features_scaled)
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            self.console.print(f"[green]✓ PCA: Reduced to {n_components} components ({explained_var:.1%} variance)[/green]")
        
        # Fit clustering model
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.user_clusters = self.model.fit_predict(user_features_scaled)
            
        elif self.method == 'agglomerative':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            self.user_clusters = self.model.fit_predict(user_features_scaled)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Add cluster labels to features
        self.user_features['cluster'] = self.user_clusters
        
        # Calculate cluster profiles
        self._calculate_cluster_profiles()
        
        # Evaluate clustering
        silhouette = silhouette_score(user_features_scaled, self.user_clusters)
        davies_bouldin = davies_bouldin_score(user_features_scaled, self.user_clusters)
        
        self.console.print(f"[green]✓ Clustering complete![/green]")
        self.console.print(f"[green]  Silhouette Score: {silhouette:.3f}[/green]")
        self.console.print(f"[green]  Davies-Bouldin Score: {davies_bouldin:.3f}[/green]")
    
    def _calculate_cluster_profiles(self):
        """Calculate statistical profiles for each cluster."""
        self.cluster_profiles = self.user_features.groupby('cluster').agg(['mean', 'std', 'count'])
        
        # Create interpretable labels
        cluster_labels = []
        for cluster_id in range(self.n_clusters):
            cluster_data = self.user_features[self.user_features['cluster'] == cluster_id]
            
            # Determine cluster characteristics
            avg_rating = cluster_data['rating_mean'].mean()
            activity = cluster_data['rating_count'].mean()
            diversity = cluster_data['movieId_nunique'].mean()
            
            # Create label
            if activity > self.user_features['rating_count'].quantile(0.75):
                activity_label = "Heavy"
            elif activity < self.user_features['rating_count'].quantile(0.25):
                activity_label = "Light"
            else:
                activity_label = "Regular"
            
            if avg_rating > 4.0:
                rating_label = "Positive"
            elif avg_rating < 3.0:
                rating_label = "Critical"
            else:
                rating_label = "Neutral"
            
            label = f"{activity_label} {rating_label} Users"
            cluster_labels.append((cluster_id, label, len(cluster_data)))
        
        self.cluster_labels = cluster_labels
    
    def display_cluster_summary(self):
        """Display summary of user clusters."""
        if self.cluster_profiles is None:
            self.console.print("[red]No clustering results available[/red]")
            return
        
        # Create summary table
        table = Table(title=f"User Segmentation Summary ({self.method.title()})", box="rounded")
        table.add_column("Cluster", style="cyan")
        table.add_column("Label", style="green")
        table.add_column("Users", justify="right")
        table.add_column("Avg Rating", justify="right")
        table.add_column("Activity", justify="right")
        table.add_column("Diversity", justify="right")
        
        for cluster_id, label, count in self.cluster_labels:
            cluster_stats = self.cluster_profiles.xs(cluster_id, level=0)
            
            table.add_row(
                str(cluster_id),
                label,
                f"{count:,}",
                f"{cluster_stats['rating_mean']['mean']:.2f}",
                f"{cluster_stats['rating_count']['mean']:.0f}",
                f"{cluster_stats['movieId_nunique']['mean']:.0f}"
            )
        
        self.console.print(table)
    
    def plot_clusters(self, save_path: Optional[Path] = None):
        """Visualize user clusters."""
        if self.pca is None:
            # Apply PCA for visualization
            pca_viz = PCA(n_components=2, random_state=self.random_state)
            features_scaled = self.scaler.transform(self.user_features.drop('cluster', axis=1))
            features_2d = pca_viz.fit_transform(features_scaled)
        else:
            # Use first 2 components
            features_scaled = self.scaler.transform(self.user_features.drop('cluster', axis=1))
            features_2d = self.pca.transform(features_scaled)[:, :2]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=self.user_clusters,
            cmap='tab10',
            alpha=0.6,
            s=30
        )
        
        # Add cluster centers
        for cluster_id in range(self.n_clusters):
            mask = self.user_clusters == cluster_id
            if mask.any():
                center = features_2d[mask].mean(axis=0)
                plt.scatter(center[0], center[1], c='black', s=200, marker='*')
                plt.annotate(f'C{cluster_id}', xy=center, xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, fontweight='bold')
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'User Segmentation - {self.method.title()} Clustering')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / f'user_segmentation_{self.method}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print("[green]✓ Cluster visualization saved[/green]")


class HierarchicalMovieClustering:
    """
    Hierarchical/Agglomerative clustering for movies based on various features.
    """
    
    def __init__(self, linkage_method: str = 'ward'):
        """
        Initialize hierarchical movie clustering.
        
        Args:
            linkage_method: 'ward', 'complete', 'average', or 'single'
        """
        self.linkage_method = linkage_method
        self.console = Console()
        
        self.scaler = StandardScaler()
        self.movie_features = None
        self.linkage_matrix = None
        self.clusters = None
        self.dendrogram_data = None
    
    def prepare_movie_features(self, ratings_df: pd.DataFrame, 
                              movies_df: pd.DataFrame,
                              use_ratings: bool = True,
                              use_genres: bool = True,
                              use_tags: bool = False,
                              tags_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare movie features for hierarchical clustering.
        
        Args:
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movie information
            use_ratings: Include rating-based features
            use_genres: Include genre features
            use_tags: Include tag-based features
            tags_df: DataFrame with tags (required if use_tags=True)
            
        Returns:
            DataFrame with movie features
        """
        self.console.print("[cyan]Preparing movie features for hierarchical clustering...[/cyan]")
        
        features_list = []
        
        # Rating-based features
        if use_ratings:
            rating_features = ratings_df.groupby('movieId').agg({
                'rating': ['count', 'mean', 'std', 'min', 'max'],
                'userId': 'nunique'
            })
            rating_features.columns = ['_'.join(col).strip() for col in rating_features.columns]
            rating_features['rating_std'] = rating_features['rating_std'].fillna(0)
            features_list.append(rating_features)
        
        # Genre features
        if use_genres:
            # One-hot encode genres
            genres_split = movies_df['genres'].str.get_dummies(sep='|')
            genres_split = genres_split.drop('(no genres listed)', axis=1, errors='ignore')
            genres_split.index = movies_df['movieId']
            features_list.append(genres_split)
        
        # Tag features
        if use_tags and tags_df is not None:
            # Create tag frequency features
            tag_features = tags_df.groupby(['movieId', 'tag']).size().unstack(fill_value=0)
            # Keep only top 50 most common tags
            top_tags = tag_features.sum().nlargest(50).index
            tag_features = tag_features[top_tags]
            features_list.append(tag_features)
        
        # Combine all features
        self.movie_features = pd.concat(features_list, axis=1).fillna(0)
        
        # Add movie metadata
        movie_info = movies_df.set_index('movieId')[['title', 'year']]
        self.movie_features = self.movie_features.join(movie_info, how='left')
        
        # Store title and year separately
        self.movie_titles = self.movie_features['title'].to_dict()
        self.movie_years = self.movie_features['year'].to_dict()
        
        # Remove non-numeric columns for clustering
        numeric_features = self.movie_features.select_dtypes(include=[np.number])
        self.movie_features_numeric = numeric_features
        
        self.console.print(f"[green]✓ Prepared {len(numeric_features.columns)} features for {len(numeric_features)} movies[/green]")
        return self.movie_features
    
    def fit_hierarchical(self, n_clusters: Optional[int] = None, 
                        distance_threshold: Optional[float] = None,
                        scale: bool = True):
        """
        Fit hierarchical clustering model.
        
        Args:
            n_clusters: Number of clusters (if None, use distance_threshold)
            distance_threshold: Distance threshold for clustering
            scale: Whether to scale features
        """
        if self.movie_features_numeric is None:
            raise ValueError("No movie features available. Run prepare_movie_features first.")
        
        self.console.print(f"[cyan]Performing hierarchical clustering with {self.linkage_method} linkage...[/cyan]")
        
        # Scale features
        if scale:
            features_scaled = self.scaler.fit_transform(self.movie_features_numeric)
        else:
            features_scaled = self.movie_features_numeric.values
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(features_scaled, method=self.linkage_method)
        
        # Determine clusters
        if n_clusters is not None:
            self.clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        elif distance_threshold is not None:
            self.clusters = fcluster(self.linkage_matrix, distance_threshold, criterion='distance')
        else:
            # Use elbow method to determine optimal number of clusters
            n_clusters = self._find_optimal_clusters(features_scaled)
            self.clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        # Add cluster labels
        self.movie_features['cluster'] = self.clusters
        
        # Calculate cluster statistics
        self._calculate_cluster_stats()
        
        self.console.print(f"[green]✓ Hierarchical clustering complete! Found {len(np.unique(self.clusters))} clusters[/green]")
    
    def _find_optimal_clusters(self, features_scaled: np.ndarray, max_clusters: int = 50) -> int:
        """Find optimal number of clusters using elbow method."""
        last = self.linkage_matrix[-max_clusters:, 2]
        last_rev = last[::-1]
        
        # Calculate acceleration (second derivative)
        acceleration = np.diff(last_rev, 2)
        
        # Find elbow point
        k = acceleration.argmax() + 2
        
        self.console.print(f"[green]✓ Optimal number of clusters: {k}[/green]")
        return k
    
    def _calculate_cluster_stats(self):
        """Calculate statistics for each cluster."""
        cluster_stats = []
        
        for cluster_id in np.unique(self.clusters):
            cluster_movies = self.movie_features[self.movie_features['cluster'] == cluster_id]
            
            # Get dominant genres
            genre_cols = [col for col in self.movie_features_numeric.columns 
                         if not any(x in col for x in ['rating', 'userId'])]
            
            if genre_cols:
                genre_sums = cluster_movies[genre_cols].sum()
                top_genres = genre_sums.nlargest(3).index.tolist()
            else:
                top_genres = []
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_movies),
                'avg_rating': cluster_movies.get('rating_mean', pd.Series([0])).mean(),
                'avg_popularity': cluster_movies.get('rating_count', pd.Series([0])).mean(),
                'dominant_genres': ', '.join(top_genres[:3])
            }
            cluster_stats.append(stats)
        
        self.cluster_stats = pd.DataFrame(cluster_stats)
    
    def plot_dendrogram(self, save_path: Optional[Path] = None, 
                       max_display: int = 50,
                       truncate_mode: str = 'level',
                       p: int = 10):
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of leaves to display
            truncate_mode: How to truncate the dendrogram
            p: Level at which to truncate
        """
        plt.figure(figsize=(15, 8))
        
        # Create dendrogram
        self.dendrogram_data = dendrogram(
            self.linkage_matrix,
            labels=self.movie_features.index.tolist(),
            leaf_rotation=90,
            leaf_font_size=8,
            truncate_mode=truncate_mode,
            p=p
        )
        
        plt.title(f'Movie Hierarchical Clustering Dendrogram ({self.linkage_method.title()} Linkage)')
        plt.xlabel('Movie ID or Cluster')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / f'movie_dendrogram_{self.linkage_method}.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print("[green]✓ Dendrogram saved[/green]")
    
    def display_cluster_summary(self):
        """Display summary of movie clusters."""
        if self.cluster_stats is None:
            self.console.print("[red]No clustering results available[/red]")
            return
        
        # Create summary table
        table = Table(title="Movie Hierarchical Clustering Summary", box="rounded")
        table.add_column("Cluster", style="cyan")
        table.add_column("Movies", justify="right")
        table.add_column("Avg Rating", justify="right")
        table.add_column("Popularity", justify="right")
        table.add_column("Dominant Genres", style="green")
        
        for _, row in self.cluster_stats.iterrows():
            table.add_row(
                str(int(row['cluster_id'])),
                f"{int(row['size']):,}",
                f"{row['avg_rating']:.2f}",
                f"{row['avg_popularity']:.0f}",
                row['dominant_genres']
            )
        
        self.console.print(table)
    
    def get_cluster_movies(self, cluster_id: int, n: int = 10) -> pd.DataFrame:
        """Get top movies from a specific cluster."""
        cluster_movies = self.movie_features[self.movie_features['cluster'] == cluster_id]
        
        # Sort by popularity (rating count)
        if 'rating_count' in cluster_movies.columns:
            cluster_movies = cluster_movies.sort_values('rating_count', ascending=False)
        
        # Get movie titles
        top_movies = []
        for movie_id in cluster_movies.index[:n]:
            title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
            year = self.movie_years.get(movie_id, "")
            top_movies.append({
                'movieId': movie_id,
                'title': title,
                'year': year,
                'rating_mean': cluster_movies.loc[movie_id].get('rating_mean', 0),
                'rating_count': cluster_movies.loc[movie_id].get('rating_count', 0)
            })
        
        return pd.DataFrame(top_movies)


class AgglomerativeComparison:
    """
    Compare different agglomerative clustering configurations.
    """
    
    def __init__(self):
        self.console = Console()
        self.results = {}
        
    def compare_linkage_methods(self, features: pd.DataFrame,
                               linkage_methods: List[str] = ['ward', 'complete', 'average', 'single'],
                               n_clusters_range: range = range(5, 21, 5)):
        """
        Compare different linkage methods and number of clusters.
        
        Args:
            features: Feature matrix
            linkage_methods: List of linkage methods to compare
            n_clusters_range: Range of cluster numbers to test
        """
        self.console.print(Panel.fit(
            "[bold cyan]Agglomerative Clustering Comparison[/bold cyan]\n"
            "Comparing different linkage methods and cluster numbers",
            border_style="cyan"
        ))
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Comparing configurations...", 
                total=len(linkage_methods) * len(n_clusters_range)
            )
            
            for linkage_method in linkage_methods:
                self.results[linkage_method] = {}
                
                # Compute linkage once for each method
                linkage_matrix = linkage(features_scaled, method=linkage_method)
                
                for n_clusters in n_clusters_range:
                    # Get clusters
                    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Calculate metrics
                    silhouette = silhouette_score(features_scaled, clusters)
                    davies_bouldin = davies_bouldin_score(features_scaled, clusters)
                    calinski = calinski_harabasz_score(features_scaled, clusters)
                    
                    self.results[linkage_method][n_clusters] = {
                        'silhouette': silhouette,
                        'davies_bouldin': davies_bouldin,
                        'calinski_harabasz': calinski,
                        'n_clusters_actual': len(np.unique(clusters))
                    }
                    
                    progress.advance(task)
        
        self._display_comparison_results()
    
    def _display_comparison_results(self):
        """Display comparison results."""
        # Find best configuration
        best_config = None
        best_silhouette = -1
        
        for linkage_method, clusters_results in self.results.items():
            for n_clusters, metrics in clusters_results.items():
                if metrics['silhouette'] > best_silhouette:
                    best_silhouette = metrics['silhouette']
                    best_config = (linkage_method, n_clusters)
        
        self.console.print(f"\n[bold green]Best Configuration:[/bold green]")
        self.console.print(f"Linkage: {best_config[0]}, Clusters: {best_config[1]}")
        self.console.print(f"Silhouette Score: {best_silhouette:.3f}")
        
        # Create comparison table
        table = Table(title="Agglomerative Clustering Results", box="rounded")
        table.add_column("Linkage", style="cyan")
        table.add_column("Clusters", justify="right")
        table.add_column("Silhouette ↑", justify="right", style="green")
        table.add_column("Davies-Bouldin ↓", justify="right", style="yellow")
        table.add_column("Calinski-Harabasz ↑", justify="right", style="blue")
        
        for linkage_method in self.results:
            for n_clusters in sorted(self.results[linkage_method].keys()):
                metrics = self.results[linkage_method][n_clusters]
                
                # Highlight best configuration
                if (linkage_method, n_clusters) == best_config:
                    linkage_display = f"[bold]{linkage_method}[/bold]"
                else:
                    linkage_display = linkage_method
                
                table.add_row(
                    linkage_display,
                    str(n_clusters),
                    f"{metrics['silhouette']:.3f}",
                    f"{metrics['davies_bouldin']:.3f}",
                    f"{metrics['calinski_harabasz']:.1f}"
                )
        
        self.console.print(table)
    
    def plot_comparison(self, save_path: Optional[Path] = None):
        """Plot comparison of different configurations."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Silhouette scores
        ax = axes[0]
        for linkage_method in self.results:
            n_clusters_list = sorted(self.results[linkage_method].keys())
            silhouettes = [self.results[linkage_method][n]['silhouette']
                          for n in n_clusters_list]
            ax.plot(n_clusters_list, silhouettes, marker='o', label=linkage_method)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Davies-Bouldin scores
        ax = axes[1]
        for linkage_method in self.results:
            n_clusters_list = sorted(self.results[linkage_method].keys())
            davies_bouldin = [self.results[linkage_method][n]['davies_bouldin']
                              for n in n_clusters_list]
            ax.plot(n_clusters_list, davies_bouldin, marker='o', label=linkage_method)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Davies-Bouldin Score')
        ax.set_title('Davies-Bouldin Score Comparison (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calinski-Harabasz scores
        ax = axes[2]
        for linkage_method in self.results:
            n_clusters_list = sorted(self.results[linkage_method].keys())
            calinski = [self.results[linkage_method][n]['calinski_harabasz']
                        for n in n_clusters_list]
            ax.plot(n_clusters_list, calinski, marker='o', label=linkage_method)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Calinski-Harabasz Score')
        ax.set_title('Calinski-Harabasz Score Comparison (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / 'agglomerative_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.console.print("[green]✓ Comparison plot saved[/green]")


def run_user_segmentation_pipeline(ratings_df: pd.DataFrame,
                                  movies_df: Optional[pd.DataFrame] = None,
                                  n_clusters: int = 8,
                                  methods: List[str] = ['kmeans', 'agglomerative']) -> Dict[str, UserSegmentation]:
    """
    Run complete user segmentation pipeline.
    
    Args:
        ratings_df: DataFrame with ratings
        movies_df: DataFrame with movie information (optional)
        n_clusters: Number of clusters
        methods: List of clustering methods to use
        
    Returns:
        Dictionary of fitted UserSegmentation objects
    """
    console.print(Panel.fit(
        "[bold cyan]User Segmentation Pipeline[/bold cyan]\n"
        f"Segmenting users into {n_clusters} clusters",
        border_style="cyan"
    ))
    
    results = {}
    
    for method in methods:
        console.print(f"\n[cyan]Running {method} clustering...[/cyan]")
        
        # Initialize segmentation
        segmentation = UserSegmentation(n_clusters=n_clusters, method=method)
        
        # Add movies data if available
        if movies_df is not None:
            segmentation.movies_df = movies_df
        
        # Prepare features
        segmentation.prepare_user_features(ratings_df)
        
        # Fit clusters
        segmentation.fit_clusters()
        
        # Display summary
        segmentation.display_cluster_summary()
        
        # Save visualization
        REPORTS_DIR.mkdir(exist_ok=True)
        segmentation.plot_clusters()
        
        results[method] = segmentation
    
    return results


def run_movie_clustering_pipeline(ratings_df: pd.DataFrame,
                                 movies_df: pd.DataFrame,
                                 tags_df: Optional[pd.DataFrame] = None,
                                 linkage_methods: List[str] = ['ward', 'complete'],
                                 n_clusters: Optional[int] = None) -> Dict[str, HierarchicalMovieClustering]:
    """
    Run complete movie clustering pipeline with hierarchical methods.
    
    Args:
        ratings_df: DataFrame with ratings
        movies_df: DataFrame with movie information
        tags_df: DataFrame with tags (optional)
        linkage_methods: List of linkage methods to use
        n_clusters: Number of clusters (if None, auto-determine)
        
    Returns:
        Dictionary of fitted HierarchicalMovieClustering objects
    """
    console.print(Panel.fit(
        "[bold cyan]Hierarchical Movie Clustering Pipeline[/bold cyan]\n"
        "Clustering movies using agglomerative methods",
        border_style="cyan"
    ))
    
    results = {}
    
    # First, compare different configurations
    console.print("\n[cyan]Comparing clustering configurations...[/cyan]")
    
    # Prepare features once
    sample_clustering = HierarchicalMovieClustering()
    features = sample_clustering.prepare_movie_features(
        ratings_df, movies_df, 
        use_tags=(tags_df is not None),
        tags_df=tags_df
    )
    
    # Run comparison
    comparison = AgglomerativeComparison()
    comparison.compare_linkage_methods(
        sample_clustering.movie_features_numeric,
        linkage_methods=linkage_methods
    )
    comparison.plot_comparison()
    
    # Run clustering with each linkage method
    for linkage_method in linkage_methods:
        console.print(f"\n[cyan]Running hierarchical clustering with {linkage_method} linkage...[/cyan]")
        
        # Initialize clustering
        clustering = HierarchicalMovieClustering(linkage_method=linkage_method)
        
        # Prepare features
        clustering.prepare_movie_features(
            ratings_df, movies_df,
            use_tags=(tags_df is not None),
            tags_df=tags_df
        )
        
        # Fit hierarchical clustering
        clustering.fit_hierarchical(n_clusters=n_clusters)
        
        # Display summary
        clustering.display_cluster_summary()
        
        # Plot dendrogram
        REPORTS_DIR.mkdir(exist_ok=True)
        clustering.plot_dendrogram()
        
        # Show sample movies from each cluster
        console.print(f"\n[bold]Sample Movies from Each Cluster ({linkage_method}):[/bold]")
        for cluster_id in range(1, min(6, len(clustering.cluster_stats) + 1)):
            console.print(f"\n[cyan]Cluster {cluster_id}:[/cyan]")
            sample_movies = clustering.get_cluster_movies(cluster_id, n=5)
            for _, movie in sample_movies.iterrows():
                console.print(f"  • {movie['title']} ({movie['year']}) - Rating: {movie['rating_mean']:.2f}")
        
        results[linkage_method] = clustering
    
    return results


def run_complete_clustering_analysis(ratings_df: pd.DataFrame,
                                    movies_df: pd.DataFrame,
                                    tags_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Run complete clustering analysis including both user and movie clustering.
    
    Args:
        ratings_df: DataFrame with ratings
        movies_df: DataFrame with movie information
        tags_df: DataFrame with tags (optional)
        
    Returns:
        Dictionary with all clustering results
    """
    console.print(Panel.fit(
        "[bold cyan]Complete Clustering Analysis[/bold cyan]\n"
        "User Segmentation + Movie Hierarchical Clustering",
        border_style="cyan"
    ))
    
    results = {
        'user_segmentation': {},
        'movie_clustering': {},
        'cross_analysis': {}
    }
    
    # User segmentation
    console.print("\n[bold yellow]Phase 1: User Segmentation[/bold yellow]")
    user_results = run_user_segmentation_pipeline(
        ratings_df, movies_df,
        n_clusters=8,
        methods=['kmeans', 'agglomerative']
    )
    results['user_segmentation'] = user_results
    
    # Movie clustering
    console.print("\n[bold yellow]Phase 2: Movie Hierarchical Clustering[/bold yellow]")
    movie_results = run_movie_clustering_pipeline(
        ratings_df, movies_df, tags_df,
        linkage_methods=['ward', 'complete', 'average'],
        n_clusters=20
    )
    results['movie_clustering'] = movie_results
    
    # Cross-analysis: How do different user segments rate different movie clusters
    console.print("\n[bold yellow]Phase 3: Cross-Analysis[/bold yellow]")
    
    # Use K-means user segmentation and Ward movie clustering for cross-analysis
    user_seg = user_results['kmeans']
    movie_clust = movie_results['ward']
    
    # Add cluster labels to dataframes
    user_clusters = user_seg.user_features[['cluster']].rename(columns={'cluster': 'user_cluster'})
    movie_clusters = movie_clust.movie_features[['cluster']].rename(columns={'cluster': 'movie_cluster'})
    
    # Merge with ratings
    cross_analysis = ratings_df.merge(
        user_clusters, left_on='userId', right_index=True
    ).merge(
        movie_clusters, left_on='movieId', right_index=True
    )
    
    # Calculate average ratings for each user cluster - movie cluster combination
    cross_ratings = cross_analysis.groupby(['user_cluster', 'movie_cluster'])['rating'].agg(['mean', 'count'])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    heatmap_data = cross_ratings['mean'].unstack(fill_value=0)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=3.5,
        cbar_kws={'label': 'Average Rating'}
    )
    plt.xlabel('Movie Cluster')
    plt.ylabel('User Cluster')
    plt.title('Cross-Analysis: User Segments vs Movie Clusters')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'user_movie_cluster_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print("[green]✓ Cross-analysis heatmap saved[/green]")
    
    # Find interesting patterns
    patterns = []
    for user_cluster in range(user_seg.n_clusters):
        for movie_cluster in range(1, movie_clust.cluster_stats['cluster_id'].max() + 1):
            if (user_cluster, movie_cluster) in cross_ratings.index:
                avg_rating = cross_ratings.loc[(user_cluster, movie_cluster), 'mean']
                count = cross_ratings.loc[(user_cluster, movie_cluster), 'count']
                
                if count >= 100:  # Significant number of ratings
                    if avg_rating >= 4.2:
                        patterns.append({
                            'user_cluster': user_cluster,
                            'movie_cluster': movie_cluster,
                            'avg_rating': avg_rating,
                            'count': count,
                            'pattern': 'Strong Preference'
                        })
                    elif avg_rating <= 2.8:
                        patterns.append({
                            'user_cluster': user_cluster,
                            'movie_cluster': movie_cluster,
                            'avg_rating': avg_rating,
                            'count': count,
                            'pattern': 'Strong Dislike'
                        })
    
    results['cross_analysis'] = {
        'cross_ratings': cross_ratings,
        'patterns': patterns
    }
    
    # Display interesting patterns
    if patterns:
        console.print("\n[bold]Interesting Patterns Found:[/bold]")
        patterns_table = Table(box="rounded")
        patterns_table.add_column("User Segment", style="cyan")
        patterns_table.add_column("Movie Cluster", style="green")
        patterns_table.add_column("Avg Rating", justify="right")
        patterns_table.add_column("# Ratings", justify="right")
        patterns_table.add_column("Pattern", style="yellow")
        
        for pattern in patterns[:10]:  # Top 10 patterns
            user_label = user_seg.cluster_labels[pattern['user_cluster']][1]
            patterns_table.add_row(
                f"{pattern['user_cluster']} ({user_label})",
                str(pattern['movie_cluster']),
                f"{pattern['avg_rating']:.2f}",
                f"{pattern['count']:,}",
                pattern['pattern']
            )
        
        console.print(patterns_table)
    
    # Save results
    results_path = PROCESSED_DATA_DIR / 'clustering_results.pkl.gz'
    with gzip.open(results_path, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    
    console.print(f"\n[green]✓ All clustering results saved to {results_path}[/green]")
    
    return results


if __name__ == "__main__":
    # Example usage
    console.print("[yellow]This module should be imported and used through the main pipeline[/yellow]")
    console.print("Example usage:")
    console.print("from movielens.models.clustering import run_complete_clustering_analysis")
    console.print("results = run_complete_clustering_analysis(ratings_df, movies_df, tags_df)")
