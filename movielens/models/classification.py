"""
Classification Models for MovieLens Dataset
Location: movielens/models/classification.py

This module implements:
- Binary classification for user satisfaction (rating >= 4.0)
- Multi-class classification for rating categories
- Genre prediction based on user patterns
- User type classification (critic, casual, enthusiast)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report, roc_auc_score,
                           roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
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
from joblib import Parallel, delayed
import multiprocessing as mp
from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
console = Console()


class RatingClassifier:
    """
    Rating classification for binary and multi-class prediction.
    """
    
    def __init__(self, task_type: str = 'binary', n_jobs: int = -1):
        """
        Initialize the rating classifier.
        
        Args:
            task_type: 'binary' (satisfied/not) or 'multiclass' (rating levels)
            n_jobs: Number of parallel jobs
        """
        self.task_type = task_type
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.console = Console()
        
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_classification_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the preprocessed classification dataset."""
        ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
        
        if not ml_datasets_path.exists():
            raise FileNotFoundError(
                "ML datasets not found. Run 'python analyze.py preprocess' first."
            )
        
        with gzip.open(ml_datasets_path, 'rb') as f:
            ml_datasets = pickle.load(f)
        
        if 'classification' not in ml_datasets or ml_datasets['classification'] is None:
            raise ValueError("Classification dataset not found in ML datasets")
        
        classification_data = ml_datasets['classification']
        X = classification_data['X']
        y = classification_data['y']
        
        self.console.print(f"[green]✓ Loaded dataset: {X.shape[0]:,} samples, {X.shape[1]} features[/green]")
        
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Prepare data for classification."""
        # Convert to appropriate target variable
        if self.task_type == 'binary':
            # Already binary (rating >= 4.0)
            self.class_names = ['Not Satisfied', 'Satisfied']
        else:
            # Convert to multi-class (5 levels)
            y = pd.cut(y, bins=[0, 1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5])
            self.class_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Display class distribution
        self._display_class_distribution()
        
    def _display_class_distribution(self):
        """Display class distribution in train/test sets."""
        train_dist = self.y_train.value_counts(normalize=True).sort_index()
        test_dist = self.y_test.value_counts(normalize=True).sort_index()
        
        table = Table(title="Class Distribution", box="rounded")
        table.add_column("Class", style="cyan")
        table.add_column("Train %", justify="right")
        table.add_column("Test %", justify="right")
        
        for idx, class_name in enumerate(self.class_names):
            train_pct = train_dist.get(idx if self.task_type == 'binary' else idx+1, 0) * 100
            test_pct = test_dist.get(idx if self.task_type == 'binary' else idx+1, 0) * 100
            table.add_row(class_name, f"{train_pct:.1f}%", f"{test_pct:.1f}%")
        
        self.console.print(table)
    
    def train_ensemble_models(self, handle_imbalance: bool = True):
        """Train multiple classification models with ensemble methods."""
        self.console.print(Panel.fit(
            f"[bold cyan]Training Ensemble Classification Models[/bold cyan]\n"
            f"Task: {self.task_type.title()} Classification",
            border_style="cyan"
        ))
        
        # Define base models
        base_models = {
            'logistic': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=self.n_jobs
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=self.n_jobs
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'svm': SVC(
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        }
        
        # Handle class imbalance if needed
        if handle_imbalance and self.task_type == 'binary':
            # Check class imbalance
            class_counts = self.y_train.value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            
            if imbalance_ratio > 1.5:
                self.console.print(f"[yellow]Class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying SMOTE.[/yellow]")
                smote = SMOTE(random_state=42, n_jobs=self.n_jobs)
                X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
            else:
                X_train_balanced, y_train_balanced = self.X_train, self.y_train
        else:
            X_train_balanced, y_train_balanced = self.X_train, self.y_train
        
        # Train each model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Training models...", total=len(base_models))
            
            for name, model in base_models.items():
                progress.update(task, description=f"[cyan]Training {name}...")
                
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics
                }
                
                progress.advance(task)
        
        # Create voting classifier
        self.console.print("\n[cyan]Creating ensemble voting classifier...[/cyan]")
        voting_models = [(name, model) for name, model in base_models.items() if name != 'svm']
        
        voting_clf = VotingClassifier(
            estimators=voting_models,
            voting='soft',
            n_jobs=self.n_jobs
        )
        
        voting_clf.fit(X_train_balanced, y_train_balanced)
        y_pred_voting = voting_clf.predict(self.X_test)
        y_pred_proba_voting = voting_clf.predict_proba(self.X_test)
        
        voting_metrics = self._calculate_metrics(self.y_test, y_pred_voting, y_pred_proba_voting)
        
        self.models['voting_ensemble'] = voting_clf
        self.results['voting_ensemble'] = {
            'model': voting_clf,
            'predictions': y_pred_voting,
            'probabilities': y_pred_proba_voting,
            'metrics': voting_metrics
        }
        
        # Find best model
        self._find_best_model()
        
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Calculate comprehensive classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate precision, recall, f1
        if self.task_type == 'binary':
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary'
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # ROC AUC for binary
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Multi-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            })
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
            metrics['per_class'] = {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1_score': f1_per_class
            }
        
        return metrics
    
    def _find_best_model(self):
        """Find the best performing model."""
        best_score = -1
        best_name = None
        
        for name, result in self.results.items():
            score = result['metrics']['f1_score']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]
        self.console.print(f"\n[green]✓ Best model: {best_name} (F1: {best_score:.3f})[/green]")
    
    def display_results(self):
        """Display comprehensive classification results."""
        # Create results table
        table = Table(title="Classification Results", box="rounded")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1 Score", justify="right", style="bold")
        
        if self.task_type == 'binary':
            table.add_column("ROC AUC", justify="right")
        
        for name, result in self.results.items():
            metrics = result['metrics']
            row = [
                name.replace('_', ' ').title(),
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}"
            ]
            
            if self.task_type == 'binary' and 'roc_auc' in metrics:
                row.append(f"{metrics['roc_auc']:.3f}")
            
            table.add_row(*row)
        
        self.console.print(table)
    
    def plot_confusion_matrices(self, save_path: Optional[Path] = None):
        """Plot confusion matrices for all models."""
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx < len(axes):
                ax = axes[idx]
                cm = result['metrics']['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=self.class_names,
                           yticklabels=self.class_names)
                ax.set_title(f'{name.replace("_", " ").title()}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
        
        # Hide empty subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / f'{self.task_type}_confusion_matrices.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print("[green]✓ Confusion matrices saved[/green]")
    
    def plot_roc_curves(self, save_path: Optional[Path] = None):
        """Plot ROC curves for binary classification."""
        if self.task_type != 'binary':
            self.console.print("[yellow]ROC curves only available for binary classification[/yellow]")
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'][:, 1])
                auc = result['metrics'].get('roc_auc', 0)
                plt.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Binary Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print("[green]✓ ROC curves saved[/green]")


class GenrePredictor:
    """
    Predict movie genres based on user rating patterns and movie features.
    """
    
    def __init__(self, n_jobs: int = -1):
        """Initialize the genre predictor."""
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.console = Console()
        
        self.genre_encoders = {}
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_genre_prediction_data(self, ratings_df: pd.DataFrame, 
                                    movies_df: pd.DataFrame,
                                    user_features: pd.DataFrame,
                                    movie_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for genre prediction."""
        self.console.print("[cyan]Preparing genre prediction data...[/cyan]")
        
        # Get all unique genres
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            if genres != '(no genres listed)':
                all_genres.update(genres.split('|'))
        
        self.genres = sorted(list(all_genres))[:10]  # Top 10 genres
        self.console.print(f"[green]✓ Found {len(self.genres)} genres for prediction[/green]")
        
        # Create binary labels for each genre
        genre_labels = pd.DataFrame(index=movies_df.index)
        for genre in self.genres:
            genre_labels[genre] = movies_df['genres'].str.contains(genre, na=False).astype(int)
        
        # Merge with movie features
        feature_matrix = movie_features.select_dtypes(include=[np.number]).fillna(0)
        
        # Add aggregated user preference features
        movie_user_prefs = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],
            'userId': 'nunique'
        })
        movie_user_prefs.columns = ['_'.join(col).strip() for col in movie_user_prefs.columns]
        
        feature_matrix = feature_matrix.join(movie_user_prefs, how='left').fillna(0)
        
        # Ensure alignment
        common_idx = feature_matrix.index.intersection(genre_labels.index)
        X = feature_matrix.loc[common_idx]
        y = genre_labels.loc[common_idx]
        
        self.console.print(f"[green]✓ Prepared {X.shape} feature matrix for genre prediction[/green]")
        
        return X, y
    
    def train_genre_predictors(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train predictors for each genre using multi-label classification."""
        self.console.print(Panel.fit(
            "[bold cyan]Training Genre Predictors[/bold cyan]\n"
            "Multi-label classification for movie genres",
            border_style="cyan"
        ))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train binary classifier for each genre
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Training genre predictors...", total=len(self.genres))
            
            for genre in self.genres:
                progress.update(task, description=f"[cyan]Training {genre} predictor...")
                
                # Use RandomForest for each genre
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=self.n_jobs
                )
                
                # Train on this genre
                clf.fit(X_train_scaled, y_train[genre])
                
                # Evaluate
                y_pred = clf.predict(X_test_scaled)
                accuracy = accuracy_score(y_test[genre], y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test[genre], y_pred, average='binary', zero_division=0
                )
                
                self.models[genre] = clf
                self.results[genre] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'feature_importance': pd.Series(
                        clf.feature_importances_,
                        index=X.columns
                    ).sort_values(ascending=False)
                }
                
                progress.advance(task)
        
        self._display_genre_results()
    
    def _display_genre_results(self):
        """Display genre prediction results."""
        table = Table(title="Genre Prediction Results", box="rounded")
        table.add_column("Genre", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1 Score", justify="right", style="bold")
        
        for genre, metrics in self.results.items():
            table.add_row(
                genre,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}"
            )
        
        self.console.print(table)
        
        # Display top features for best performing genre
        best_genre = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        self.console.print(f"\n[bold]Top Features for {best_genre} Prediction:[/bold]")
        top_features = self.results[best_genre]['feature_importance'].head(10)
        
        for feature, importance in top_features.items():
            self.console.print(f"  • {feature}: {importance:.3f}")


class UserTypeClassifier:
    """
    Classify users into behavioral types (critic, casual, enthusiast, etc.)
    """
    
    def __init__(self, n_clusters: int = 5):
        """Initialize user type classifier."""
        self.n_clusters = n_clusters
        self.console = Console()
        self.model = None
        self.user_types = None
        self.scaler = StandardScaler()
        
    def create_user_behavior_features(self, ratings_df: pd.DataFrame,
                                    user_features: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features for user type classification."""
        self.console.print("[cyan]Creating user behavioral features...[/cyan]")
        
        # Additional behavioral features
        user_behavior = ratings_df.groupby('userId').agg({
            'rating': [
                'mean',  # Average rating given
                'std',   # Rating variance
                'count', # Total ratings
                lambda x: (x >= 4).sum() / len(x),  # Positive rating ratio
                lambda x: (x <= 2).sum() / len(x),  # Negative rating ratio
            ],
            'movieId': 'nunique',  # Number of unique movies rated
            'timestamp': [
                lambda x: (x.max() - x.min()).days,  # Activity span in days
                lambda x: len(x) / ((x.max() - x.min()).days + 1)  # Ratings per day
            ]
        })
        
        # Flatten column names
        user_behavior.columns = [
            'avg_rating', 'rating_std', 'total_ratings', 
            'positive_ratio', 'negative_ratio', 'unique_movies',
            'activity_span_days', 'ratings_per_day'
        ]
        
        # Add genre diversity
        if 'genres' in ratings_df.columns:
            genre_diversity = ratings_df.groupby('userId')['genres'].apply(
                lambda x: len(set('|'.join(x.dropna()).split('|')))
            )
            user_behavior['genre_diversity'] = genre_diversity
        
        # Combine with existing features
        combined_features = user_features.join(user_behavior, how='left').fillna(0)
        
        return combined_features
    
    def classify_user_types(self, user_features: pd.DataFrame) -> pd.Series:
        """Classify users into behavioral types."""
        self.console.print("[cyan]Classifying user types...[/cyan]")
        
        # Scale features
        user_features_scaled = self.scaler.fit_transform(
            user_features.select_dtypes(include=[np.number]).fillna(0)
        )
        
        # Use K-means for initial clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(user_features_scaled)
        
        # Analyze clusters to assign meaningful types
        self.user_types = self._assign_user_types(user_features, cluster_labels)
        
        return self.user_types
    
    def _assign_user_types(self, features: pd.DataFrame, 
                          cluster_labels: np.ndarray) -> pd.Series:
        """Assign meaningful user type labels based on cluster characteristics."""
        features['cluster'] = cluster_labels
        
        # Analyze each cluster
        cluster_profiles = features.groupby('cluster').agg({
            'avg_rating': 'mean',
            'rating_std': 'mean',
            'total_ratings': 'mean',
            'positive_ratio': 'mean',
            'unique_movies': 'mean'
        })
        
        # Define user types based on characteristics
        user_type_map = {}
        
        for cluster_id in range(self.n_clusters):
            profile = cluster_profiles.loc[cluster_id]
            
            if profile['avg_rating'] < 3.0 and profile['rating_std'] < 1.0:
                user_type_map[cluster_id] = 'Critic'
            elif profile['total_ratings'] > features['total_ratings'].quantile(0.75):
                user_type_map[cluster_id] = 'Enthusiast'
            elif profile['total_ratings'] < features['total_ratings'].quantile(0.25):
                user_type_map[cluster_id] = 'Casual Viewer'
            elif profile['positive_ratio'] > 0.7:
                user_type_map[cluster_id] = 'Positive Rater'
            else:
                user_type_map[cluster_id] = 'Regular User'
        
        # Map cluster labels to user types
        user_types = pd.Series(
            [user_type_map[label] for label in cluster_labels],
            index=features.index
        )
        
        self._display_user_type_distribution(user_types)
        
        return user_types
    
    def _display_user_type_distribution(self, user_types: pd.Series):
        """Display distribution of user types."""
        distribution = user_types.value_counts()
        
        table = Table(title="User Type Distribution", box="rounded")
        table.add_column("User Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        total_users = len(user_types)
        for user_type, count in distribution.items():
            percentage = (count / total_users) * 100
            table.add_row(user_type, f"{count:,}", f"{percentage:.1f}%")
        
        self.console.print(table)


def run_classification_pipeline(sample_size: Optional[int] = None) -> Dict:
    """
    Run complete classification pipeline.
    
    Args:
        sample_size: Optional sample size for faster testing
        
    Returns:
        Dictionary with all classification results
    """
    console.print(Panel.fit(
        "[bold cyan]Classification Pipeline[/bold cyan]\n"
        "Binary, Multi-class, Genre Prediction, and User Type Classification",
        border_style="cyan"
    ))
    
    results = {}
    
    # Binary classification
    console.print("\n[bold yellow]Phase 1: Binary Classification (Satisfaction Prediction)[/bold yellow]")
    binary_clf = RatingClassifier(task_type='binary')
    
    try:
        X, y = binary_clf.load_classification_dataset()
        
        if sample_size and sample_size < len(X):
            console.print(f"[yellow]Sampling {sample_size:,} examples for faster testing[/yellow]")
            Sampleindices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[Sampleindices]
            y = y.iloc[Sampleindices]

        binary_clf.prepare_data(X, y)
        binary_clf.train_ensemble_models()
        binary_clf.display_results()
        
        # Save visualizations
        REPORTS_DIR.mkdir(exist_ok=True)
        binary_clf.plot_confusion_matrices()
        binary_clf.plot_roc_curves()
        
        results['binary_classification'] = binary_clf
        
    except Exception as e:
        console.print(f"[red]Binary classification failed: {e}[/red]")
        results['binary_classification'] = None
    
    # Multi-class classification
    console.print("\n[bold yellow]Phase 2: Multi-class Classification (Rating Levels)[/bold yellow]")
    multiclass_clf = RatingClassifier(task_type='multiclass')
    
    try:
        # Load raw ratings for multi-class
        ratings_path = PROCESSED_DATA_DIR / "ratings.parquet"
        if ratings_path.exists():
            ratings_df = pd.read_parquet(ratings_path)
            
            # Use same features but different target
            if 'binary_classification' in results and results['binary_classification']:
                X = results['binary_classification'].X_train.append(
                    results['binary_classification'].X_test
                )
                y = ratings_df.loc[X.index, 'rating']
                
                multiclass_clf.prepare_data(X, y)
                multiclass_clf.train_ensemble_models(handle_imbalance=False)
                multiclass_clf.display_results()
                multiclass_clf.plot_confusion_matrices()
                
                results['multiclass_classification'] = multiclass_clf
        else:
            console.print("[yellow]Ratings data not found for multi-class classification[/yellow]")
            results['multiclass_classification'] = None
            
    except Exception as e:
        console.print(f"[red]Multi-class classification failed: {e}[/red]")
        results['multiclass_classification'] = None
    
    # Genre prediction
    console.print("\n[bold yellow]Phase 3: Genre Prediction[/bold yellow]")
    
    try:
        # Load necessary data
        movies_path = PROCESSED_DATA_DIR / "movies.parquet"
        user_features_path = PROCESSED_DATA_DIR / "user_features.parquet"
        movie_features_path = PROCESSED_DATA_DIR / "movie_features.parquet"
        
        if all(p.exists() for p in [movies_path, user_features_path, movie_features_path]):
            movies_df = pd.read_parquet(movies_path)
            user_features = pd.read_parquet(user_features_path)
            movie_features = pd.read_parquet(movie_features_path)
            
            if 'ratings_df' not in locals():
                ratings_df = pd.read_parquet(PROCESSED_DATA_DIR / "ratings.parquet")
            
            genre_predictor = GenrePredictor()
            X_genre, y_genre = genre_predictor.prepare_genre_prediction_data(
                ratings_df, movies_df, user_features, movie_features
            )
            
            if sample_size and len(X_genre) > sample_size:
                sample_idx = np.random.choice(len(X_genre), sample_size, replace=False)
                X_genre = X_genre.iloc[sample_idx]
                y_genre = y_genre.iloc[sample_idx]
            
            genre_predictor.train_genre_predictors(X_genre, y_genre)
            results['genre_prediction'] = genre_predictor
        else:
            console.print("[yellow]Required data files not found for genre prediction[/yellow]")
            results['genre_prediction'] = None
            
    except Exception as e:
        console.print(f"[red]Genre prediction failed: {e}[/red]")
        results['genre_prediction'] = None
    
    # User type classification
    console.print("\n[bold yellow]Phase 4: User Type Classification[/bold yellow]")
    
    try:
        if 'user_features' in locals() and 'ratings_df' in locals():
            user_classifier = UserTypeClassifier(n_clusters=5)
            
            # Create enhanced behavioral features
            behavioral_features = user_classifier.create_user_behavior_features(
                ratings_df, user_features
            )
            
            # Classify users
            user_types = user_classifier.classify_user_types(behavioral_features)
            
            results['user_type_classification'] = {
                'classifier': user_classifier,
                'user_types': user_types,
                'features': behavioral_features
            }
        else:
            console.print("[yellow]Required data not available for user type classification[/yellow]")
            results['user_type_classification'] = None
            
    except Exception as e:
        console.print(f"[red]User type classification failed: {e}[/red]")
        results['user_type_classification'] = None
    
    # Save all results
    console.print("\n[cyan]Saving classification results...[/cyan]")
    
    try:
        results_path = PROCESSED_DATA_DIR / "classification_results.pkl.gz"
        with gzip.open(results_path, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        console.print(f"[green]✓ Results saved to {results_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")
    
    # Display final summary
    _display_classification_summary(results)
    
    return results


def _display_classification_summary(results: Dict):
    """Display summary of all classification tasks."""
    console = Console()
    
    summary_table = Table(title="Classification Pipeline Summary", box="rounded")
    summary_table.add_column("Task", style="cyan")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Best Model", style="green")
    summary_table.add_column("Performance", justify="right")
    
    # Binary classification
    if results.get('binary_classification'):
        binary_clf = results['binary_classification']
        best_model = max(binary_clf.results.items(), key=lambda x: x[1]['metrics']['f1_score'])
        summary_table.add_row(
            "Binary Classification",
            "[green]✓[/green]",
            best_model[0].replace('_', ' ').title(),
            f"F1: {best_model[1]['metrics']['f1_score']:.3f}"
        )
    else:
        summary_table.add_row("Binary Classification", "[red]✗[/red]", "N/A", "N/A")
    
    # Multi-class classification
    if results.get('multiclass_classification'):
        multi_clf = results['multiclass_classification']
        best_model = max(multi_clf.results.items(), key=lambda x: x[1]['metrics']['f1_score'])
        summary_table.add_row(
            "Multi-class Classification",
            "[green]✓[/green]",
            best_model[0].replace('_', ' ').title(),
            f"F1: {best_model[1]['metrics']['f1_score']:.3f}"
        )
    else:
        summary_table.add_row("Multi-class Classification", "[red]✗[/red]", "N/A", "N/A")
    
    # Genre prediction
    if results.get('genre_prediction'):
        genre_pred = results['genre_prediction']
        avg_f1 = np.mean([r['f1_score'] for r in genre_pred.results.values()])
        summary_table.add_row(
            "Genre Prediction",
            "[green]✓[/green]",
            "Random Forest (per genre)",
            f"Avg F1: {avg_f1:.3f}"
        )
    else:
        summary_table.add_row("Genre Prediction", "[red]✗[/red]", "N/A", "N/A")
    
    # User type classification
    if results.get('user_type_classification'):
        summary_table.add_row(
            "User Type Classification",
            "[green]✓[/green]",
            "K-Means + Rule-based",
            f"{len(results['user_type_classification']['user_types'].unique())} types"
        )
    else:
        summary_table.add_row("User Type Classification", "[red]✗[/red]", "N/A", "N/A")
    
    console.print(summary_table)


if __name__ == "__main__":
    # Example usage
    console.print("[yellow]Running classification pipeline...[/yellow]")
    results = run_classification_pipeline(sample_size=10000)  # Use sample for testing