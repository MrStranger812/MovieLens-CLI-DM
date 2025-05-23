import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
import click
from .cleaner import DataCleaner
from .transformer import DataTransformer
from ..config import *

class PreprocessingPipeline:
    """Complete preprocessing pipeline combining cleaning and transformation."""
    
    def __init__(self):
        self.console = Console()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.processed_data = {}
        self.feature_sets = {}
        
    def run_full_pipeline(self, create_sparse_matrices: bool = True, 
                     apply_pca: bool = True, save_results: bool = True,
                     use_cache: bool = True) -> Dict:
        """Run the complete preprocessing pipeline with caching support."""
        
        self.console.print(Panel.fit(
            "[bold blue]Starting Complete Preprocessing Pipeline[/bold blue]\n"
            "This will clean, transform, and prepare all data for ML algorithms",
            border_style="blue"
        ))
        
        results = {}
        
        # Check for cached results first
        cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
        if use_cache and cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                self.console.print("[yellow]Found cached preprocessing results![/yellow]")
                
                # Ask user if they want to use cache
                if click.confirm("Use cached preprocessing results? (faster)", default=True):
                    self.console.print("[green]✓ Using cached results[/green]")
                    self._display_pipeline_summary(cached_results)
                    self.processed_data = cached_results
                    return cached_results
            except Exception as e:
                self.console.print(f"[yellow]Cache loading failed: {e}. Running fresh preprocessing...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=False,
            console=self.console
        ) as progress:
            
            # Step 1: Data Loading and Basic Cleaning
            task = progress.add_task("[cyan]Phase 1: Data Loading & Cleaning...", total=7)
            
            # Load raw data
            ratings_df, movies_df, tags_df = self.cleaner.load_data()
            progress.advance(task)
            
            # Validate data
            self.cleaner.validate_data()
            progress.advance(task)
            
            # Clean all datasets
            cleaned_ratings, cleaned_movies, cleaned_tags = self.cleaner.clean_all()
            results['cleaned_data'] = {
                'ratings': cleaned_ratings,
                'movies': cleaned_movies,
                'tags': cleaned_tags
            }
            progress.advance(task)
            
            # Step 2: Advanced Data Cleaning
            progress.update(task, description="[cyan]Phase 2: Advanced Cleaning...")
            
            # Handle missing values
            cleaned_ratings = self.transformer.handle_missing_values(cleaned_ratings)
            cleaned_movies = self.transformer.handle_missing_values(cleaned_movies)
            progress.advance(task)
            
            # Remove duplicates
            cleaned_ratings = self.transformer.remove_duplicates(
                cleaned_ratings, subset=['userId', 'movieId', 'timestamp']
            )
            progress.advance(task)
            
            # Detect and handle outliers in ratings
            cleaned_ratings = self.transformer.detect_outliers(
                cleaned_ratings, columns=['rating'], method='iqr'
            )
            progress.advance(task)
            
            # Step 3: Feature Engineering
            progress.update(task, description="[cyan]Phase 3: Feature Engineering...")
            
            # Create user features
            user_features = self.transformer.create_user_features(cleaned_ratings)
            
            # Create movie features
            movie_features = self.transformer.create_movie_features(cleaned_ratings, cleaned_movies)
            
            # Create comprehensive feature matrix
            feature_matrix = self._create_comprehensive_features(
                cleaned_ratings, user_features, movie_features
            )
            
            results['feature_engineering'] = {
                'user_features': user_features,
                'movie_features': movie_features,
                'feature_matrix': feature_matrix
            }
            progress.advance(task)
            
            # Step 4: Categorical Encoding
            progress.update(task, description="[cyan]Phase 4: Encoding & Normalization...")
            
            # Encode categorical features
            categorical_columns = ['day_of_week']
            if 'day_of_week' in feature_matrix.columns:
                feature_matrix = self.transformer.encode_categorical_features(
                    feature_matrix, categorical_columns, method='label'
                )
            
            # Step 5: Feature Normalization
            numerical_columns = [
                'rating', 'year', 'month', 'hour',
                'user_count', 'user_mean', 'user_std',
                'movie_count', 'movie_mean', 'movie_std'
            ]
            available_numerical = [col for col in numerical_columns if col in feature_matrix.columns]
            
            if available_numerical:
                feature_matrix = self.transformer.normalize_features(
                    feature_matrix, available_numerical, method='standard'
                )
            
            results['encoded_data'] = feature_matrix
            
            # Step 6: Dimensionality Reduction (Optional)
            if apply_pca:
                progress.update(task, description="[cyan]Phase 5: Dimensionality Reduction...")
                
                # Select numerical features for PCA
                numerical_features = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numerical_features) > 10:  # Only apply PCA if we have enough features
                    pca_features, pca_summary = self.transformer.apply_pca(
                        feature_matrix, numerical_features, variance_threshold=0.95
                    )
                    results['pca_features'] = pca_features
                    results['pca_summary'] = pca_summary
            
            # Step 7: Create Specialized Data Structures
            if create_sparse_matrices:
                progress.update(task, description="[cyan]Phase 6: Creating Specialized Matrices...")
                
                # Check if user-item matrix already exists
                matrix_cache_file = PROCESSED_DATA_DIR / "user_item_matrix.npz"
                mappings_cache_file = PROCESSED_DATA_DIR / "user_item_mappings.pkl"
                
                if use_cache and matrix_cache_file.exists() and mappings_cache_file.exists():
                    try:
                        from scipy import sparse
                        import pickle
                        
                        # Load cached matrix and mappings
                        user_item_matrix = sparse.load_npz(matrix_cache_file)
                        with open(mappings_cache_file, 'rb') as f:
                            user_mapping, movie_mapping = pickle.load(f)
                        
                        self.console.print("[green]✓ Loaded cached user-item matrix[/green]")
                        sparsity = 1 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
                        self.console.print(f"[green]✓ Matrix: {user_item_matrix.shape[0]}x{user_item_matrix.shape[1]}, Sparsity: {sparsity:.4f}[/green]")
                        
                    except Exception as e:
                        self.console.print(f"[yellow]Cache loading failed: {e}. Computing matrix...[/yellow]")
                        # Fallback to computing
                        user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(
                            cleaned_ratings, sparse=True
                        )
                        
                        # Save to cache
                        if save_results:
                            sparse.save_npz(matrix_cache_file, user_item_matrix)
                            with open(mappings_cache_file, 'wb') as f:
                                pickle.dump((user_mapping, movie_mapping), f)
                else:
                    # Create user-item matrix
                    user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(
                        cleaned_ratings, sparse=True
                    )
                    
                    # Save to cache
                    if save_results:
                        from scipy import sparse
                        import pickle
                        sparse.save_npz(matrix_cache_file, user_item_matrix)
                        with open(mappings_cache_file, 'wb') as f:
                            pickle.dump((user_mapping, movie_mapping), f)
                
                results['collaborative_filtering'] = {
                    'user_item_matrix': user_item_matrix,
                    'user_mapping': user_mapping,
                    'movie_mapping': movie_mapping
                }
            
            # Step 8: Prepare Dataset Splits for Different ML Tasks
            progress.update(task, description="[cyan]Phase 7: Preparing ML-Ready Datasets...")
            
            # Prepare datasets for different ML tasks
            ml_datasets = self._prepare_ml_datasets(results)
            results['ml_ready_datasets'] = ml_datasets
            
            # Step 9: Save Results (Optional)
            if save_results:
                self._save_processed_data(results)
                
                # Save cache
                try:
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(results, f)
                    self.console.print(f"[green]✓ Results cached for future use[/green]")
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
            
            progress.update(task, description="[green]✓ Pipeline Complete!")
        
        # Display summary
        self._display_pipeline_summary(results)
        
        self.processed_data = results
        return results
    
    def _create_comprehensive_features(self, ratings_df: pd.DataFrame, 
                                     user_features: pd.DataFrame, 
                                     movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive feature matrix by merging all features."""
        
        # Start with ratings data
        feature_matrix = ratings_df.copy()
        
        # Merge user features
        feature_matrix = feature_matrix.merge(
            user_features, left_on='userId', right_index=True, how='left'
        )
        
        # Merge movie features
        feature_matrix = feature_matrix.merge(
            movie_features, left_on='movieId', right_index=True, how='left'
        )
        
        # Create interaction features
        interaction_pairs = [
            ('user_mean', 'movie_mean'),
            ('user_count', 'movie_count'),
            ('year', 'movie_age')
        ]
        
        available_pairs = [
            (f1, f2) for f1, f2 in interaction_pairs 
            if f1 in feature_matrix.columns and f2 in feature_matrix.columns
        ]
        
        if available_pairs:
            feature_matrix = self.transformer.create_interaction_features(
                feature_matrix, available_pairs
            )
        
        return feature_matrix
    
    def _prepare_ml_datasets(self, results: Dict) -> Dict:
        """Prepare datasets optimized for different ML tasks."""
        
        ml_datasets = {}
        
        # Get the main feature matrix
        feature_matrix = results['encoded_data']
        
        # 1. Regression Dataset (predicting ratings)
        # Exclude ID columns and target variable from features
        excluded_columns = ['rating', 'userId', 'movieId', 'timestamp']
        regression_features = [col for col in feature_matrix.columns 
                            if col not in excluded_columns]
        
        if regression_features:
            # Handle different data types properly
            X_regression = feature_matrix[regression_features].copy()
            
            # Fill missing values based on column type
            for col in X_regression.columns:
                if X_regression[col].dtype.name == 'category':
                    # For categorical columns, fill with most frequent category
                    mode_value = X_regression[col].mode()
                    if len(mode_value) > 0:
                        X_regression[col] = X_regression[col].fillna(mode_value[0])
                elif X_regression[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
                    # For numerical columns, fill with 0
                    X_regression[col] = X_regression[col].fillna(0)
                else:
                    # For other types, fill with 0 or appropriate default
                    try:
                        X_regression[col] = X_regression[col].fillna(0)
                    except:
                        # If fillna(0) fails, use forward fill then backward fill
                        X_regression[col] = X_regression[col].fillna(method='ffill').fillna(method='bfill')
            
            ml_datasets['regression'] = {
                'X': X_regression,
                'y': feature_matrix['rating'],
                'feature_names': regression_features
            }
        
        # 2. Classification Dataset (like/dislike prediction)
        if regression_features:
            ml_datasets['classification'] = {
                'X': X_regression.copy(),  # Use the same cleaned features
                'y': (feature_matrix['rating'] >= 4.0).astype(int),  # Binary: like (1) vs dislike (0)
                'feature_names': regression_features
            }
        
        # 3. Clustering Dataset (user profiles)
        if 'user_features' in results['feature_engineering']:
            user_features = results['feature_engineering']['user_features']
            clustering_features = user_features.select_dtypes(include=[np.number]).fillna(0)
            
            ml_datasets['clustering_users'] = {
                'X': clustering_features,
                'feature_names': clustering_features.columns.tolist()
            }
        
        # 4. Clustering Dataset (movie profiles)
        if 'movie_features' in results['feature_engineering']:
            movie_features = results['feature_engineering']['movie_features']
            movie_clustering_features = movie_features.select_dtypes(include=[np.number]).fillna(0)
            
            ml_datasets['clustering_movies'] = {
                'X': movie_clustering_features,
                'feature_names': movie_clustering_features.columns.tolist()
            }
        
        # 5. Association Rules Dataset (transaction format)
        ratings_df = results['cleaned_data']['ratings']
        
        # Create transaction data for association rules (movies rated >= 4.0 by each user)
        positive_ratings = ratings_df[ratings_df['rating'] >= 4.0]
        transactions = positive_ratings.groupby('userId')['movieId'].apply(list).tolist()
        
        ml_datasets['association_rules'] = {
            'transactions': transactions,
            'movies_df': results['cleaned_data']['movies']
        }
        
        return ml_datasets
    
    def _save_processed_data(self, results: Dict):
        """Save all processed data to the processed directory."""
        
        # Ensure directory exists
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
        # Save cleaned datasets
        if 'cleaned_data' in results:
            for dataset_name, df in results['cleaned_data'].items():
                filepath = PROCESSED_DATA_DIR / f"{dataset_name}_cleaned.parquet"
                df.to_parquet(filepath, index=False)
        
        # Save feature datasets
        if 'feature_engineering' in results:
            for feature_name, df in results['feature_engineering'].items():
                if isinstance(df, pd.DataFrame):
                    filepath = PROCESSED_DATA_DIR / f"{feature_name}.parquet"
                    df.to_parquet(filepath, index=True)
        
        # Save ML-ready datasets
        if 'ml_ready_datasets' in results:
            import pickle
            
            # Save as pickle for complex structures
            ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl"
            with open(ml_datasets_path, 'wb') as f:
                pickle.dump(results['ml_ready_datasets'], f)
        
        # Save transformers
        transformers_path = PROCESSED_DATA_DIR / "transformers.pkl"
        self.transformer.save_transformers(transformers_path)
        
        self.console.print(f"[green]✓ All processed data saved to {PROCESSED_DATA_DIR}[/green]")
    
    def _display_pipeline_summary(self, results: Dict):
        """Display a comprehensive summary of the preprocessing pipeline."""
        
        self.console.print("\n" + "="*60)
        self.console.print(Panel.fit(
            "[bold green]Preprocessing Pipeline Complete![/bold green]",
            border_style="green"
        ))
        
        # Data summary table
        summary_table = Table(title="Processed Data Summary")
        summary_table.add_column("Dataset", style="cyan")
        summary_table.add_column("Records", justify="right")
        summary_table.add_column("Features", justify="right")
        summary_table.add_column("Memory (MB)", justify="right")
        
        if 'cleaned_data' in results:
            for name, df in results['cleaned_data'].items():
                if isinstance(df, pd.DataFrame):
                    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                    summary_table.add_row(
                        name.title(),
                        f"{len(df):,}",
                        f"{len(df.columns)}",
                        f"{memory_mb:.2f}"
                    )
        
        self.console.print(summary_table)
        
        # Feature engineering summary
        if 'feature_engineering' in results:
            fe_table = Table(title="Feature Engineering Summary")
            fe_table.add_column("Feature Set", style="cyan")
            fe_table.add_column("Features Created", justify="right")
            
            for name, df in results['feature_engineering'].items():
                if isinstance(df, pd.DataFrame):
                    fe_table.add_row(name.replace('_', ' ').title(), f"{len(df.columns)}")
            
            self.console.print(fe_table)
        
        # ML datasets summary
        if 'ml_ready_datasets' in results:
            ml_table = Table(title="ML-Ready Datasets")
            ml_table.add_column("ML Task", style="cyan")
            ml_table.add_column("Samples", justify="right")
            ml_table.add_column("Features", justify="right")
            ml_table.add_column("Target Type", style="yellow")
            
            for task_name, dataset in results['ml_ready_datasets'].items():
                if 'X' in dataset and hasattr(dataset['X'], 'shape'):
                    samples, features = dataset['X'].shape
                    
                    if 'y' in dataset:
                        if hasattr(dataset['y'], 'dtype'):
                            target_type = str(dataset['y'].dtype)
                        else:
                            target_type = "N/A"
                    else:
                        target_type = "Unsupervised"
                    
                    ml_table.add_row(
                        task_name.replace('_', ' ').title(),
                        f"{samples:,}",
                        f"{features}",
                        target_type
                    )
                elif task_name == 'association_rules':
                    ml_table.add_row(
                        "Association Rules",
                        f"{len(dataset['transactions']):,}",
                        "Transactions",
                        "Itemsets"
                    )
            
            self.console.print(ml_table)
        
        # Transformation summary
        transformation_summary = self.transformer.get_transformation_summary()
        if transformation_summary['transformations_applied']:
            self.console.print("\n[bold]Transformations Applied:[/bold]")
            for i, transformation in enumerate(transformation_summary['transformations_applied'], 1):
                self.console.print(f"  {i}. {transformation}")
        
        self.console.print(f"\n[bold green]📁 Processed data saved to: {PROCESSED_DATA_DIR}[/bold green]")
        self.console.print("[dim]You can now proceed to implement ML algorithms![/dim]")
    
    def load_processed_data(self) -> Dict:
        """Load previously processed data."""
        
        if not PROCESSED_DATA_DIR.exists():
            self.console.print("[red]No processed data found. Run the preprocessing pipeline first.[/red]")
            return {}
        
        # Load ML datasets
        ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl"
        if ml_datasets_path.exists():
            import pickle
            with open(ml_datasets_path, 'rb') as f:
                ml_datasets = pickle.load(f)
            
            self.console.print("[green]✓ Loaded ML-ready datasets[/green]")
            return {'ml_ready_datasets': ml_datasets}
        
        return {}
    
    def get_dataset_for_task(self, task: str) -> Dict:
        """Get preprocessed dataset for a specific ML task."""
        
        if not self.processed_data and not self.load_processed_data():
            self.console.print("[yellow]No processed data available. Run preprocessing first.[/yellow]")
            return {}
        
        available_tasks = [
            'regression', 'classification', 'clustering_users', 
            'clustering_movies', 'association_rules'
        ]
        
        if task not in available_tasks:
            self.console.print(f"[red]Invalid task. Available tasks: {available_tasks}[/red]")
            return {}
        
        if 'ml_ready_datasets' in self.processed_data:
            return self.processed_data['ml_ready_datasets'].get(task, {})
        
        return {}