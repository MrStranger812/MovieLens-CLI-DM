import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import click
from .cleaner import DataCleaner
from .transformer import DataTransformer
from ..config import *
import pickle
import gzip
import psutil
import gc
from scipy import sparse
import traceback
from joblib import Parallel, delayed
import multiprocessing as mp
import time

class PreprocessingPipeline:
    """Complete preprocessing pipeline with 4GB VRAM optimization."""

    def __init__(self, n_jobs: int = None):
        self.console = Console()
        self.cleaner = DataCleaner()
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.transformer = DataTransformer(n_jobs=self.n_jobs)
        self.processed_data = {}
        self.feature_sets = {}
        
        self.console.print(f"[green]✓ Pipeline initialized with {self.n_jobs} parallel workers[/green]")

    def load_cached_results(self) -> Optional[Dict]:
        """Load previously processed data."""
        cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
        ml_datasets_gz_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"

        if ml_datasets_gz_path.exists():
            self.console.print(f"[yellow]Found ML datasets: {ml_datasets_gz_path}[/yellow]")
            try:
                with gzip.open(ml_datasets_gz_path, 'rb') as f:
                    ml_datasets = pickle.load(f)
                self.console.print("[green]✓ Loaded ML-ready datasets[/green]")
                self.processed_data = {'ml_ready_datasets': ml_datasets}
                return self.processed_data
            except Exception as e:
                self.console.print(f"[red]Error loading {ml_datasets_gz_path}: {e}[/red]")
                return None
        else:
            self.console.print("[red]No processed data found.[/red]")
            return None

    def run_full_pipeline_with_monitoring(self,
                                        create_sparse_matrices: bool = True,
                                        apply_pca: bool = False,  # Disabled for 4GB VRAM
                                        save_results: bool = True,
                                        use_cache: bool = True,
                                        memory_limit_gb: float = 16.0,
                                        validate_steps: bool = True,
                                        large_dataset_threshold: int = 2_000_000,
                                        sparse_tfidf_threshold: int = 1_000_000,
                                        batch_size: int = 200_000) -> Optional[Dict]:
        """
        Optimized preprocessing pipeline for 4GB VRAM.
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3

        self.console.print(Panel.fit(
            "[bold blue]Starting 4GB VRAM Optimized Pipeline[/bold blue]\n"
            f"System Memory: {psutil.virtual_memory().percent:.1f}% used\n"
            f"Process Memory: {initial_memory:.2f} GB\n"
            f"Parallel Workers: {self.n_jobs}\n"
            f"Focus: Regression & Classification only",
            border_style="blue"
        ))

        pipeline_metrics = {
            'start_time': pd.Timestamp.now(), 
            'steps_completed': [], 
            'memory_usage': [],
            'validation_results': {}, 
            'feature_counts': {}, 
            'data_stats': {},
            'parallel_workers': self.n_jobs,
            'batch_size': batch_size
        }
        results = {}

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeElapsedColumn(),
            console=self.console, 
            refresh_per_second=2
        ) as progress:
            main_task = progress.add_task("[bold cyan]Pipeline Progress", total=7)
            step_name = "Initialization"
            
            try:
                # ============ STEP 1: Data Loading ============
                step_name = "Data Loading"
                self._log_step_start(step_name)
                progress.update(main_task, description="Loading data...")
                
                ratings_df, movies_df, tags_df = self.cleaner.load_data()
                
                # Optimize data types immediately
                ratings_df = self.transformer.optimize_dtypes(ratings_df)
                movies_df = self.transformer.optimize_dtypes(movies_df)
                
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)

                # ============ STEP 2: Data Cleaning ============
                step_name = "Data Cleaning"
                self._log_step_start(step_name)
                progress.update(main_task, description="Cleaning data...")
                
                cleaned_ratings = self.cleaner.clean_ratings(save=False, optimize_memory=True)
                cleaned_movies = self.cleaner.clean_movies(save=False)
                
                # Skip tags for memory optimization
                cleaned_tags = None
                
                # Basic data quality
                cleaned_ratings = self.transformer.handle_missing_values(cleaned_ratings)
                cleaned_ratings = self.transformer.remove_duplicates(
                    cleaned_ratings, subset=['userId', 'movieId', 'timestamp']
                )
                
                results['cleaned_data'] = {
                    'ratings': cleaned_ratings, 
                    'movies': cleaned_movies, 
                    'tags': cleaned_tags
                }
                
                # Memory check
                current_memory = process.memory_info().rss / 1024**3
                if current_memory > 8:  # Conservative for 4GB VRAM
                    self._free_memory()
                
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)

                # ============ STEP 3: User Features (Simple) ============
                step_name = "User Features"
                self._log_step_start(step_name)
                progress.update(main_task, description="Creating user features...")
                
                user_features = self.transformer.create_user_features(cleaned_ratings)
                pipeline_metrics['feature_counts']['user_features'] = len(user_features.columns)
                
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)

                # ============ STEP 4: Movie Features (Simple) ============
                step_name = "Movie Features"
                self._log_step_start(step_name)
                progress.update(main_task, description="Creating movie features...")
                
                movie_features = self.transformer.create_movie_features(cleaned_ratings, cleaned_movies)
                pipeline_metrics['feature_counts']['movie_features'] = len(movie_features.columns)
                
                results['feature_engineering'] = {
                    'user_features': user_features, 
                    'movie_features': movie_features
                }
                
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)

                # ============ STEP 5: Sparse Matrix ============
                if create_sparse_matrices:
                    step_name = "Sparse Matrix"
                    self._log_step_start(step_name)
                    progress.update(main_task, description="Creating sparse matrix...")
                    
                    user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(
                        cleaned_ratings, sparse=True
                    )
                    
                    results['collaborative_filtering'] = {
                        'user_item_matrix': user_item_matrix, 
                        'user_mapping': user_mapping, 
                        'movie_mapping': movie_mapping
                    }
                    
                    self._log_step_complete(step_name, pipeline_metrics, process)
                
                progress.advance(main_task)

                # ============ STEP 6: ML Datasets (Regression & Classification Only) ============
                step_name = "ML Datasets"
                self._log_step_start(step_name)
                progress.update(main_task, description="Creating ML datasets...")
                
                ml_datasets = self._create_ml_datasets_optimized(
                    cleaned_ratings, user_features, movie_features
                )
                
                results['ml_ready_datasets'] = ml_datasets
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)

                # ============ STEP 7: Save Results ============
                if save_results:
                    step_name = "Saving Results"
                    self._log_step_start(step_name)
                    progress.update(main_task, description="Saving results...")
                    self._save_results_optimized(results)
                    self._log_step_complete(step_name, pipeline_metrics, process)
                
                progress.advance(main_task)

            except Exception as e:
                self._log_step_error(f"Pipeline failed at {step_name}", e)
                return None

            finally:
                pipeline_metrics['end_time'] = pd.Timestamp.now()
                pipeline_metrics['total_time'] = (pipeline_metrics['end_time'] - pipeline_metrics['start_time']).total_seconds()

        self._display_summary(results, pipeline_metrics)
        self.processed_data = results
        return results

    def _create_ml_datasets_optimized(self, ratings_df: pd.DataFrame, 
                                    user_features: pd.DataFrame,
                                    movie_features: pd.DataFrame) -> Dict:
        """Create ML datasets optimized for 4GB VRAM."""
        self.console.print("[cyan]Creating ML datasets (memory-optimized)...[/cyan]")
        
        ml_datasets = {}
        
        try:
            # Sample for memory efficiency
            sample_size = min(1_000_000, len(ratings_df))
            if len(ratings_df) > sample_size:
                self.console.print(f"[yellow]Sampling {sample_size:,} ratings for ML datasets[/yellow]")
                ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
            else:
                ratings_sample = ratings_df.copy()
            
            # Prepare feature matrix efficiently
            self.console.print("[cyan]Merging features...[/cyan]")
            
            # Start with ratings
            feature_matrix = ratings_sample[['userId', 'movieId', 'rating']].copy()
            
            # Select only numeric features
            user_numeric = user_features.select_dtypes(include=[np.number]).columns.tolist()
            movie_numeric = movie_features.select_dtypes(include=[np.number]).columns.tolist()
            
            # Limit features if too many
            max_features_per_type = 50
            if len(user_numeric) > max_features_per_type:
                user_numeric = user_numeric[:max_features_per_type]
            if len(movie_numeric) > max_features_per_type:
                movie_numeric = movie_numeric[:max_features_per_type]
            
            # Merge user features
            user_features_subset = user_features[user_numeric].copy()
            user_features_subset.columns = [f'user_{col}' for col in user_features_subset.columns]
            
            feature_matrix = feature_matrix.merge(
                user_features_subset,
                left_on='userId',
                right_index=True,
                how='left'
            )
            
            # Merge movie features
            movie_features_subset = movie_features[movie_numeric].copy()
            movie_features_subset.columns = [f'movie_{col}' for col in movie_features_subset.columns]
            
            feature_matrix = feature_matrix.merge(
                movie_features_subset,
                left_on='movieId',
                right_index=True,
                how='left'
            )
            
            # Fill NaN values
            feature_matrix = feature_matrix.fillna(0)
            
            # Get feature columns
            feature_cols = [col for col in feature_matrix.columns 
                           if col not in ['userId', 'movieId', 'rating']]
            
            self.console.print(f"[green]Total features: {len(feature_cols)}[/green]")
            
            # Extract features and targets
            X = feature_matrix[feature_cols].values.astype(np.float32)
            y_regression = feature_matrix['rating'].values.astype(np.float32)
            y_classification = (feature_matrix['rating'] >= 4.0).astype(np.int32)
            
            # Create regression dataset
            ml_datasets['regression'] = {
                'X': X,
                'y': y_regression,
                'feature_names': feature_cols,
                'n_samples': len(X),
                'n_features': len(feature_cols)
            }
            self.console.print(f"[green]✓ Regression dataset: {X.shape}[/green]")
            
            # Create classification dataset
            ml_datasets['classification'] = {
                'X': X,
                'y': y_classification,
                'feature_names': feature_cols,
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'class_balance': {
                    0: int((y_classification == 0).sum()),
                    1: int((y_classification == 1).sum())
                }
            }
            self.console.print(f"[green]✓ Classification dataset: {X.shape}[/green]")
            
            # Add minimal placeholders for other tasks
            ml_datasets['clustering_users'] = None
            ml_datasets['clustering_movies'] = None
            ml_datasets['association_rules'] = None
            
            # Clean up
            del feature_matrix
            gc.collect()
            
        except Exception as e:
            self.console.print(f"[red]Error creating ML datasets: {e}[/red]")
            traceback.print_exc()
            
            # Return None for all tasks on error
            for task in ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']:
                ml_datasets[task] = None
        
        return ml_datasets

    def _save_results_optimized(self, results: Dict):
        """Save results with memory optimization."""
        self.console.print("\n[cyan]Saving results...[/cyan]")
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
        # Save features as parquet
        if 'feature_engineering' in results:
            for name, df in results['feature_engineering'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    filepath = PROCESSED_DATA_DIR / f"{name}.parquet"
                    df.to_parquet(filepath, compression='snappy', index=True)
                    self.console.print(f"[green]✓ Saved {name}[/green]")
        
        # Save ML datasets
        if 'ml_ready_datasets' in results:
            ml_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
            with gzip.open(ml_path, 'wb', compresslevel=6) as f:
                pickle.dump(results['ml_ready_datasets'], f, protocol=4)
            self.console.print(f"[green]✓ Saved ML datasets[/green]")
        
        # Save sparse matrix
        if 'collaborative_filtering' in results:
            matrix_path = PROCESSED_DATA_DIR / "user_item_matrix.npz"
            sparse.save_npz(matrix_path, results['collaborative_filtering']['user_item_matrix'])
            
            mappings_path = PROCESSED_DATA_DIR / "user_item_mappings.pkl"
            with open(mappings_path, 'wb') as f:
                pickle.dump((
                    results['collaborative_filtering']['user_mapping'],
                    results['collaborative_filtering']['movie_mapping']
                ), f)
            self.console.print(f"[green]✓ Saved sparse matrix[/green]")

    def _display_summary(self, results: Dict, metrics: Dict):
        """Display pipeline summary."""
        self.console.print("\n" + "="*60)
        self.console.print(Panel.fit(
            "[bold green]Pipeline Complete![/bold green]", 
            border_style="green"
        ))

        # Execution metrics
        total_time = metrics.get('total_time', 0)
        self.console.print(f"\nTotal time: {int(total_time // 60)}m {int(total_time % 60)}s")
        
        # ML datasets summary
        if 'ml_ready_datasets' in results:
            ml_table = Table(title="ML Datasets", box=box.ROUNDED)
            ml_table.add_column("Task", style="cyan")
            ml_table.add_column("Status", justify="center")
            ml_table.add_column("Samples", justify="right")
            ml_table.add_column("Features", justify="right")
            
            for task, data in results['ml_ready_datasets'].items():
                if data and isinstance(data, dict) and 'X' in data:
                    ml_table.add_row(
                        task.title(),
                        "[green]Ready[/green]",
                        f"{data['n_samples']:,}",
                        str(data['n_features'])
                    )
                else:
                    ml_table.add_row(
                        task.title(),
                        "[yellow]Skipped[/yellow]",
                        "—",
                        "—"
                    )
            
            self.console.print(ml_table)

        self.console.print(f"\n[bold green]✅ Data saved to {PROCESSED_DATA_DIR}[/bold green]")

    def _log_step_start(self, step_name: str):
        """Log the start of a pipeline step."""
        self.console.print(f"\n[bold cyan]▶ {step_name}[/bold cyan]")

    def _log_step_complete(self, step_name: str, metrics: Dict, process: psutil.Process):
        """Log the completion of a pipeline step."""
        metrics['steps_completed'].append(step_name)
        current_memory = process.memory_info().rss / 1024**3
        metrics['memory_usage'].append(current_memory)
        self.console.print(f"[green]✓ {step_name} complete (Memory: {current_memory:.2f}GB)[/green]")

    def _log_step_error(self, step_name: str, error: Exception):
        """Log an error in a pipeline step."""
        self.console.print(f"[red]✗ Error in {step_name}: {error}[/red]")
        traceback.print_exc()

    def _free_memory(self):
        """Free up memory by garbage collection."""
        gc.collect()
        gc.collect()
        self.console.print("[yellow]♻ Memory cleanup performed[/yellow]")

    # Simplified methods for 4GB VRAM
    def get_dataset_for_task(self, task: str) -> Dict:
        """Get preprocessed dataset for a specific ML task."""
        if not self.processed_data:
            self.load_cached_results()

        if not self.processed_data:
            self.console.print("[yellow]No processed data available. Run 'preprocess' first.[/yellow]")
            return {}

        ml_datasets = self.processed_data.get('ml_ready_datasets', {})
        return ml_datasets.get(task, {})

# Commented out methods not needed for 4GB VRAM optimization:
"""
    def _create_comprehensive_features(): # Too memory intensive
    def _validate_raw_data(): # Skip validation for speed
    def _create_feature_matrix_optimized(): # Replaced with simpler version
    def _validate_ml_datasets(): # Skip validation
    def _validate_cache(): # Simplified
    def run_benchmark_subset(): # Not needed
    def get_performance_stats(): # Not needed
    def run_optimized_pipeline(): # Using main pipeline instead
"""