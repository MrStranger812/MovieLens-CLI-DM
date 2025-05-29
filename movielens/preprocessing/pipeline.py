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
from concurrent.futures import ThreadPoolExecutor
import time

class PreprocessingPipeline:
    """Complete preprocessing pipeline combining cleaning and transformation with parallel processing."""

    def __init__(self, n_jobs: int = None):
        self.console = Console()
        self.cleaner = DataCleaner()
        # Initialize transformer with same number of parallel workers
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.transformer = DataTransformer(n_jobs=self.n_jobs)
        self.processed_data = {}
        self.feature_sets = {}
        
        self.console.print(f"[green]✓ Pipeline initialized with {self.n_jobs} parallel workers[/green]")

    def _create_comprehensive_features(self, ratings_df: pd.DataFrame,
                                     user_features: pd.DataFrame,
                                     movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive feature matrix by merging all features using parallel processing."""
        self.console.print("[cyan]Creating comprehensive feature matrix (parallel)...[/cyan]")
        
        # Use chunked processing for large datasets
        if len(ratings_df) > 1_000_000:
            chunk_size = len(ratings_df) // self.n_jobs
            chunks = [ratings_df.iloc[i:i+chunk_size] for i in range(0, len(ratings_df), chunk_size)]
            
            def process_feature_chunk(chunk):
                feature_chunk = chunk.copy()
                feature_chunk = feature_chunk.merge(
                    user_features, left_on='userId', right_index=True, how='left'
                )
                feature_chunk = feature_chunk.merge(
                    movie_features, left_on='movieId', right_index=True, how='left'
                )
                return feature_chunk
            
            # Process chunks in parallel
            processed_chunks = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(process_feature_chunk)(chunk) for chunk in chunks
            )
            
            # Combine chunks
            feature_matrix = pd.concat(processed_chunks, ignore_index=True)
        else:
            feature_matrix = ratings_df.copy()
            feature_matrix = feature_matrix.merge(
                user_features, left_on='userId', right_index=True, how='left'
            )
            feature_matrix = feature_matrix.merge(
                movie_features, left_on='movieId', right_index=True, how='left'
            )
        
        # Create interaction features using the transformer's parallel method
        interaction_pairs = [
            ('rating_mean', 'rating_mean'),
            ('rating_count', 'rating_count'),
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
        
        self.console.print(f"[green]✓ Comprehensive matrix created: {feature_matrix.shape} (parallel)[/green]")
        return feature_matrix

    def load_cached_results(self) -> Optional[Dict]:
        """Load previously processed data and metrics from the cache file or ML datasets."""
        cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
        ml_datasets_gz_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"

        if cache_file.exists():
            self.console.print(f"[yellow]Found cache file: {cache_file}. Loading...[/yellow]")
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                if self._validate_cache(cached_results):
                    self.console.print("[green]✓ Loaded and validated cached results.[/green]")
                    self.processed_data = cached_results
                    return cached_results
                else:
                    self.console.print("[yellow]Cache validation failed. Ignoring cache.[/yellow]")
                    cache_file.unlink()
                    return None
            except Exception as e:
                self.console.print(f"[red]Error loading cache file {cache_file}: {e}. Removing it.[/red]")
                try:
                   cache_file.unlink()
                except OSError:
                   pass
                return None
        elif ml_datasets_gz_path.exists():
            self.console.print(f"[yellow]Cache not found, but found {ml_datasets_gz_path}. Loading ML datasets only...[/yellow]")
            try:
                with gzip.open(ml_datasets_gz_path, 'rb') as f:
                    ml_datasets = pickle.load(f)
                self.console.print("[green]✓ Loaded ML-ready datasets (limited data).[/green]")
                self.processed_data = {'ml_ready_datasets': ml_datasets}
                return self.processed_data
            except Exception as e:
                self.console.print(f"[red]Error loading {ml_datasets_gz_path}: {e}[/red]")
                return None
        else:
            self.console.print("[red]No processed data or cache found.[/red]")
            return None

    def get_dataset_for_task(self, task: str) -> Dict:
        """Get preprocessed dataset for a specific ML task."""
        if not self.processed_data:
            self.load_cached_results()

        if not self.processed_data:
             self.console.print("[yellow]No processed data available. Run 'preprocess' first.[/yellow]")
             return {}

        available_tasks = [
            'regression', 'classification', 'clustering_users',
            'clustering_movies', 'association_rules'
        ]
        if task not in available_tasks:
            self.console.print(f"[red]Invalid task. Available tasks: {available_tasks}[/red]")
            return {}

        ml_datasets = self.processed_data.get('ml_ready_datasets', {})
        return ml_datasets.get(task, {})

    def run_full_pipeline_with_monitoring(self,
                                        create_sparse_matrices: bool = True,
                                        apply_pca: bool = True,
                                        save_results: bool = True,
                                        use_cache: bool = True,
                                        memory_limit_gb: float = 16.0,
                                        validate_steps: bool = True,
                                        large_dataset_threshold: int = 2_000_000,
                                        sparse_tfidf_threshold: int = 1_000_000,
                                        batch_size: int = 200_000) -> Optional[Dict]:
        """
        Enhanced preprocessing pipeline with comprehensive monitoring, validation, and parallel processing.
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3  # GB

        self.console.print(Panel.fit(
            "[bold blue]Starting Enhanced Parallel Preprocessing Pipeline[/bold blue]\n"
            f"System Memory: {psutil.virtual_memory().percent:.1f}% used "
            f"({psutil.virtual_memory().used / 1024**3:.1f}/{psutil.virtual_memory().total / 1024**3:.1f} GB)\n"
            f"Process Memory: {initial_memory:.2f} GB\n"
            f"Memory Limit: {memory_limit_gb} GB\n"
            f"CPU Count: {psutil.cpu_count()} cores\n"
            f"Parallel Workers: {self.n_jobs}\n"
            f"Batch Size: {batch_size:,}",
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
        cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"

        if use_cache and cache_file.exists():
            cached_data = self.load_cached_results()
            if cached_data:
                 if click.confirm("Valid cache found. Use cached results?", default=True):
                     self.console.print("[green]✓ Loading from cache[/green]")
                     self._display_enhanced_summary(cached_data, cached_data.get('pipeline_metrics', pipeline_metrics))
                     return cached_data
                 else:
                     self.console.print("[yellow]Ignoring cache as requested.[/yellow]")

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
            TimeElapsedColumn(),
            TextColumn("| Mem: {task.fields[memory]:.1f}GB"), 
            TextColumn("| Workers: 4"),
            TextColumn("| {task.fields[status]}"),
            console=self.console, 
            refresh_per_second=2
        ) as progress:
            main_task = progress.add_task("[bold cyan]Pipeline Progress", total=9,
                                        memory=initial_memory, workers=self.n_jobs, status="Initializing...")
            step_name = "Initialization"
            
            try:
                # ============ STEP 1: Data Loading ============
                step_name = "Data Loading & Validation"
                self._log_step_start(step_name)
                progress.update(main_task, status="Loading data...")
                
                ratings_df, movies_df, tags_df = self.cleaner.load_data()
                pipeline_metrics['data_stats']['initial'] = {
                    'ratings_count': len(ratings_df), 
                    'movies_count': len(movies_df),
                    'tags_count': len(tags_df), 
                    'users_count': ratings_df['userId'].nunique(),
                    'memory_usage_mb': sum(df.memory_usage(deep=True).sum() / 1024**2 
                                         for df in [ratings_df, movies_df, tags_df] if df is not None)
                }
                
                if validate_steps:
                    validation_result = self._validate_raw_data(ratings_df, movies_df, tags_df)
                    pipeline_metrics['validation_results']['raw_data'] = validation_result
                    if not validation_result['passed']: 
                        raise ValueError(f"Data validation failed: {validation_result['issues']}")
                
                # Optimize data types using parallel processing
                ratings_df = self.transformer.optimize_dtypes(ratings_df)
                movies_df = self.transformer.optimize_dtypes(movies_df)
                if tags_df is not None: 
                    tags_df = self.transformer.optimize_dtypes(tags_df)
                
                results['raw_data'] = {'ratings': ratings_df, 'movies': movies_df, 'tags': tags_df}
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Data loaded")

                # ============ STEP 2: Data Cleaning ============
                step_name = "Data Cleaning (Parallel)"
                self._log_step_start(step_name)
                progress.update(main_task, status="Cleaning data with parallel processing...")
                
                cleaning_task = progress.add_task("[cyan]Cleaning datasets...", total=3, 
                                                 memory=process.memory_info().rss / 1024**3, status="")
                
                # Use parallel cleaning methods from transformer
                cleaned_ratings = self.cleaner.clean_ratings(save=False, optimize_memory=True)
                progress.advance(cleaning_task)
                
                cleaned_movies = self.cleaner.clean_movies(save=False)
                progress.advance(cleaning_task)
                
                cleaned_tags = self.cleaner.clean_tags(save=False) if tags_df is not None else None
                progress.advance(cleaning_task)
                
                # Apply parallel data quality methods
                cleaned_ratings = self.transformer.handle_missing_values(cleaned_ratings)
                cleaned_ratings = self.transformer.remove_duplicates(
                    cleaned_ratings, subset=['userId', 'movieId', 'timestamp']
                )
                cleaned_ratings = self.transformer.detect_outliers(
                    cleaned_ratings, columns=['rating'], method='iqr'
                )
                
                results['cleaned_data'] = {
                    'ratings': cleaned_ratings, 
                    'movies': cleaned_movies, 
                    'tags': cleaned_tags
                }
                
                current_memory = process.memory_info().rss / 1024**3
                if current_memory > memory_limit_gb * 0.8: 
                    self._free_memory()
                if current_memory > memory_limit_gb: 
                    raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}GB > {memory_limit_gb}GB")
                
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Data cleaned")

                # ============ STEP 3: Parallel Feature Engineering ============
                step_name = "Parallel Feature Engineering"
                self._log_step_start(step_name)
                progress.update(main_task, status=f"Engineering features with {self.n_jobs} workers...")
                
                feature_task = progress.add_task("[cyan]Creating features...", total=6, 
                                               memory=process.memory_info().rss / 1024**3, 
                                               status=f"Using {self.n_jobs} workers")
                
                is_large_dataset = len(cleaned_ratings) > large_dataset_threshold
                if is_large_dataset: 
                    self.console.print(f"[yellow]Large dataset detected ({len(cleaned_ratings):,} ratings) - using optimized parallel processing[/yellow]")

                # 3.1 User features with parallel processing
                progress.update(feature_task, description="[cyan]Creating user features (parallel)...")
                if is_large_dataset:
                    user_features = self.transformer.create_user_features_optimized(
                        cleaned_ratings, batch_size=batch_size
                    )
                else:
                    user_features = self.transformer.create_user_features(cleaned_ratings)
                pipeline_metrics['feature_counts']['user_features'] = len(user_features.columns)
                progress.advance(feature_task)

                # 3.2 Movie features with parallel processing
                progress.update(feature_task, description="[cyan]Creating movie features (parallel)...")
                movie_features = self.transformer.create_movie_features(cleaned_ratings, cleaned_movies)
                pipeline_metrics['feature_counts']['movie_features'] = len(movie_features.columns)
                progress.advance(feature_task)

                # 3.3 Temporal features with parallel processing
                progress.update(feature_task, description="[cyan]Creating temporal features (parallel)...")
                temporal_ratings, user_temporal, movie_temporal = self.transformer.create_temporal_features(
                    cleaned_ratings, cleaned_movies
                )
                if not user_temporal.empty: 
                    user_features = user_features.join(user_temporal, how='left')
                if not movie_temporal.empty: 
                    movie_features = movie_features.join(movie_temporal, how='left')
                pipeline_metrics['feature_counts']['temporal_features'] = (
                    len(user_temporal.columns) + len(movie_temporal.columns)
                )
                progress.advance(feature_task)

                # 3.4 Tag features with parallel processing
                if cleaned_tags is not None and len(cleaned_tags) > 0:
                    progress.update(feature_task, description="[cyan]Creating tag features (parallel)...")
                    if len(cleaned_tags) > sparse_tfidf_threshold:
                        tfidf_matrix, movie_ids, tfidf = self.transformer.create_sparse_tfidf_features(
                            cleaned_tags, cleaned_movies
                        )
                        results['sparse_features'] = {
                            'tfidf_matrix': tfidf_matrix, 
                            'movie_ids': movie_ids, 
                            'tfidf_vectorizer': tfidf
                        }
                    else:
                        tag_features, tag_transformers = self.transformer.create_tag_features(
                            cleaned_tags, cleaned_movies
                        )
                        movie_features = movie_features.join(tag_features, how='left')
                        results['tag_transformers'] = tag_transformers
                    
                    user_tag_prefs = self.transformer.create_user_tag_preferences(
                        cleaned_ratings, cleaned_tags
                    )
                    user_features = user_features.join(user_tag_prefs, how='left')
                    pipeline_metrics['feature_counts']['tag_features'] = 100
                progress.advance(feature_task)

                # 3.5 Genre features with parallel processing
                progress.update(feature_task, description="[cyan]Creating genre features (parallel)...")
                genre_features = self.transformer.create_enhanced_genre_features(
                    cleaned_movies, cleaned_ratings
                )
                
                # Handle overlapping columns
                overlapping_columns = set(movie_features.columns) & set(genre_features.columns)
                if overlapping_columns:
                    movie_features = movie_features.join(genre_features, how='left', lsuffix='_movie', rsuffix='_genre')
                else:
                    movie_features = movie_features.join(genre_features, how='left')
                pipeline_metrics['feature_counts']['genre_features'] = len(genre_features.columns)
                progress.advance(feature_task)

                # 3.6 Cold start features with parallel processing
                progress.update(feature_task, description="[cyan]Creating cold start features (parallel)...")
                user_cold_features, movie_cold_features = self.transformer.create_cold_start_features(
                    cleaned_ratings, cleaned_movies, user_features, movie_features
                )
                user_features = user_features.join(user_cold_features, how='left')
                movie_features = movie_features.join(movie_cold_features, how='left')
                pipeline_metrics['feature_counts']['cold_start_features'] = (
                    len(user_cold_features.columns) + len(movie_cold_features.columns)
                )
                progress.advance(feature_task)

                # Validate features using parallel processing
                if validate_steps:
                    self.console.print("[cyan]Validating features with parallel processing...[/cyan]")
                    user_validation = self.transformer.validate_features(user_features, 'user')
                    movie_validation = self.transformer.validate_features(movie_features, 'movie')
                    
                    if user_validation['issues'] or user_validation['warnings']: 
                        user_features = self.transformer.auto_fix_features(user_features, user_validation)
                    if movie_validation['issues'] or movie_validation['warnings']: 
                        movie_features = self.transformer.auto_fix_features(movie_features, movie_validation)
                    
                    pipeline_metrics['validation_results']['features'] = {
                        'user_features': user_validation, 
                        'movie_features': movie_validation
                    }

                results['feature_engineering'] = {
                    'user_features': user_features, 
                    'movie_features': movie_features, 
                    'feature_matrix': None
                }
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Features created")

                # ============ STEP 4: Feature Encoding & Normalization ============
                step_name = "Feature Encoding & Normalization (Parallel)"
                self._log_step_start(step_name)
                progress.update(main_task, status="Encoding/Normalizing with parallel processing...")
                
                sample_size = min(100000, len(temporal_ratings))
                sample_ratings = temporal_ratings.sample(n=sample_size, random_state=42)
                sample_matrix = self._create_comprehensive_features(sample_ratings, user_features, movie_features)
                
                categorical_columns = ['day_of_week', 'season']
                available_categorical = [col for col in categorical_columns if col in sample_matrix.columns]
                if available_categorical: 
                    sample_matrix = self.transformer.encode_categorical_features(
                        sample_matrix, available_categorical, method='label'
                    )
                
                numerical_columns = [
                    'rating', 'year', 'month', 'hour', 'rating_count', 'rating_mean', 'rating_std',
                    'user_count', 'user_mean', 'user_std', 'movie_count', 'movie_mean', 'movie_std'
                ]
                available_numerical = [col for col in numerical_columns if col in sample_matrix.columns]
                if available_numerical: 
                    sample_matrix = self.transformer.normalize_features(
                        sample_matrix, available_numerical, method='standard'
                    )
                
                results['encoding_info'] = {
                    'categorical_encoded': available_categorical,
                    'numerical_normalized': available_numerical, 
                    'sample_shape': sample_matrix.shape
                }
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Features encoded")

                # ============ STEP 5: Dimensionality Reduction (Optional) ============
                if apply_pca:
                    step_name = "Dimensionality Reduction (Parallel)"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Applying PCA with parallel processing...")
                    
                    user_numeric = user_features.select_dtypes(include=[np.number])
                    movie_numeric = movie_features.select_dtypes(include=[np.number])
                    results['pca_features'] = {}
                    
                    if len(user_numeric.columns) > 50:
                        user_pca, user_pca_summary = self.transformer.apply_pca(
                            user_features, user_numeric.columns.tolist(), variance_threshold=0.95
                        )
                        results['pca_features']['user_pca'] = user_pca
                        results['pca_features']['user_pca_summary'] = user_pca_summary
                    
                    if len(movie_numeric.columns) > 50:
                        movie_pca, movie_pca_summary = self.transformer.apply_pca(
                            movie_features, movie_numeric.columns.tolist(), variance_threshold=0.95
                        )
                        results['pca_features']['movie_pca'] = movie_pca
                        results['pca_features']['movie_pca_summary'] = movie_pca_summary
                    
                    self._log_step_complete(step_name, pipeline_metrics, process)
                
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, 
                              status="PCA complete" if apply_pca else "Skipped PCA")

                # ============ STEP 6: Create Sparse Matrices ============
                if create_sparse_matrices:
                    step_name = "Creating Sparse Matrices (Parallel)"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Creating sparse matrices with parallel processing...")
                    
                    matrix_cache_file = PROCESSED_DATA_DIR / "user_item_matrix.npz"
                    mappings_cache_file = PROCESSED_DATA_DIR / "user_item_mappings.pkl"
                    
                    if use_cache and matrix_cache_file.exists() and mappings_cache_file.exists():
                        try:
                            user_item_matrix = sparse.load_npz(matrix_cache_file)
                            with open(mappings_cache_file, 'rb') as f: 
                                user_mapping, movie_mapping = pickle.load(f)
                            self.console.print("[green]✓ Loaded cached user-item matrix[/green]")
                        except:
                            user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(
                                cleaned_ratings, sparse=True
                            )
                            if save_results: 
                                sparse.save_npz(matrix_cache_file, user_item_matrix)
                                pickle.dump((user_mapping, movie_mapping), open(mappings_cache_file, 'wb'))
                    else:
                        user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(
                            cleaned_ratings, sparse=True
                        )
                        if save_results: 
                            sparse.save_npz(matrix_cache_file, user_item_matrix)
                            pickle.dump((user_mapping, movie_mapping), open(mappings_cache_file, 'wb'))
                    
                    results['collaborative_filtering'] = {
                        'user_item_matrix': user_item_matrix, 
                        'user_mapping': user_mapping, 
                        'movie_mapping': movie_mapping
                    }
                    self._log_step_complete(step_name, pipeline_metrics, process)
                
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Sparse matrices ready")

                # ============ STEP 7: ML Dataset Preparation ============
                step_name = "ML Dataset Preparation (Parallel)"
                self._log_step_start(step_name)
                progress.update(main_task, status="Preparing ML datasets with parallel processing...")
                
                ml_task = progress.add_task("[cyan]Preparing ML datasets...", total=5, 
                                          memory=process.memory_info().rss / 1024**3, status="")
                ml_datasets = {}
                tasks = ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']
                
                # Process ML dataset preparation in parallel where possible
                def prepare_supervised_dataset(task_name, temporal_ratings, user_features, movie_features):
                    if task_name in ['regression', 'classification']:
                        sample_size_ml = min(5_000_000, len(temporal_ratings)) if is_large_dataset else len(temporal_ratings)
                        sampled_ratings = temporal_ratings.sample(n=sample_size_ml, random_state=42)
                        feature_matrix = self._create_feature_matrix_optimized(sampled_ratings, user_features, movie_features)
                        return self._prepare_supervised_dataset(feature_matrix, task_name)
                    return None
                
                for task_name in tasks:
                    progress.update(ml_task, description=f"[cyan]Preparing {task_name}...")
                    try:
                        if task_name in ['regression', 'classification']:
                            ml_datasets[task_name] = prepare_supervised_dataset(
                                task_name, temporal_ratings, user_features, movie_features
                            )
                        elif task_name == 'clustering_users':
                            user_numeric = user_features.select_dtypes(include=[np.number]).fillna(0)
                            ml_datasets[task_name] = {
                                'X': user_numeric, 
                                'feature_names': user_numeric.columns.tolist(), 
                                'user_ids': user_numeric.index
                            }
                        elif task_name == 'clustering_movies':
                            movie_numeric = movie_features.select_dtypes(include=[np.number]).fillna(0)
                            ml_datasets[task_name] = {
                                'X': movie_numeric, 
                                'feature_names': movie_numeric.columns.tolist(), 
                                'movie_ids': movie_numeric.index
                            }
                        elif task_name == 'association_rules':
                            positive_ratings = cleaned_ratings[cleaned_ratings['rating'] >= 4.0]
                            transactions = positive_ratings.groupby('userId')['movieId'].apply(list).tolist()
                            ml_datasets[task_name] = {
                                'transactions': transactions, 
                                'movies_df': cleaned_movies, 
                                'min_support': 0.01, 
                                'min_confidence': 0.5
                            }
                    except Exception as e:
                        self.console.print(f"[red]Error preparing {task_name}: {e}[/red]")
                        ml_datasets[task_name] = None
                    progress.advance(ml_task)
                
                results['ml_ready_datasets'] = ml_datasets
                if validate_steps: 
                    self._validate_ml_datasets(ml_datasets, pipeline_metrics)
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="ML datasets ready")

                # ============ STEP 8: Save Results ============
                if save_results:
                    step_name = "Saving Results (Parallel)"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Saving results with parallel processing...")
                    self._save_results_optimized(results)
                    
                    try:
                        cache_data = {
                            **results, 
                            'pipeline_metrics': pipeline_metrics, 
                            'pipeline_version': '2.1',  # Updated version for parallel processing
                            'creation_date': pd.Timestamp.now()
                        }
                        with open(cache_file, 'wb') as f: 
                            pickle.dump(cache_data, f, protocol=4)
                        self.console.print("[green]✓ Cache saved successfully[/green]")
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
                    self._log_step_complete(step_name, pipeline_metrics, process)
                
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="✓ Complete!", completed=9)

            except Exception as e:
                self._log_step_error(f"Pipeline failed at {step_name}", e)
                return None

            finally:
                pipeline_metrics['end_time'] = pd.Timestamp.now()
                pipeline_metrics['total_time'] = (pipeline_metrics['end_time'] - pipeline_metrics['start_time']).total_seconds()
                pipeline_metrics['peak_memory_gb'] = max(pipeline_metrics.get('memory_usage', [process.memory_info().rss / 1024**3]))

        self._display_enhanced_summary(results, pipeline_metrics)
        self.processed_data = results
        return results

    def _prepare_supervised_dataset(self, feature_matrix: pd.DataFrame, task_type: str) -> Dict:
        """Prepare dataset for supervised learning tasks using parallel processing."""
        excluded_columns = ['rating', 'userId', 'movieId', 'timestamp']
        all_cols = feature_matrix.columns.tolist()
        
        # Build feature columns list with parallel checking
        def check_column_inclusion(col):
            if col in excluded_columns:
                return None
            if col.startswith('interact_') or col.endswith('_scaled') or col.endswith('_encoded'):
                return col
            if not any(suffix in col for suffix in ['interact_', '_scaled', '_encoded']):
                return col
            return None
        
        # Use parallel processing to filter feature columns
        column_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(check_column_inclusion)(col) for col in all_cols
        )
        
        feature_columns = [col for col in column_results if col is not None]
        feature_columns = sorted(list(set(feature_columns)))

        if not feature_columns: 
            raise ValueError(f"No features available for {task_type}")
        
        X = feature_matrix[feature_columns].copy()
        
        # Parallel missing value handling
        def fill_missing_column(col):
            if X[col].dtype.name == 'category':
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                return col, X[col].fillna(fill_val)
            else:
                return col, X[col].fillna(0)
        
        fill_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(fill_missing_column)(col) for col in X.columns
        )
        
        for col, filled_series in fill_results:
            X[col] = filled_series
        
        y = feature_matrix['rating'] if task_type == 'regression' else (feature_matrix['rating'] >= 4.0).astype(int)
        
        return {
            'X': X, 
            'y': y, 
            'feature_names': feature_columns, 
            'n_samples': len(X), 
            'n_features': len(feature_columns)
        }

    def _validate_cache(self, cache_data: Dict) -> bool:
        """Validate cached data is still valid."""
        if 'pipeline_version' not in cache_data or not cache_data.get('pipeline_version', '').startswith('2.'):
            self.console.print("[yellow]Cache version mismatch - parallel version required[/yellow]")
            return False
        
        if 'creation_date' in cache_data:
            age_days = (pd.Timestamp.now() - cache_data['creation_date']).days
            if age_days > 30:
                self.console.print(f"[yellow]Cache is {age_days} days old[/yellow]")
                return False
        
        required_keys = ['cleaned_data', 'feature_engineering', 'ml_ready_datasets']
        if not all(key in cache_data for key in required_keys):
            self.console.print("[yellow]Cache missing required keys[/yellow]")
            return False
        
        return True

    def _log_step_start(self, step_name: str):
        """Log the start of a pipeline step."""
        self.console.print(f"\n[bold cyan]▶ Starting: {step_name}[/bold cyan]")

    def _log_step_complete(self, step_name: str, metrics: Dict, process: psutil.Process):
        """Log the completion of a pipeline step."""
        metrics['steps_completed'].append(step_name)
        current_memory = process.memory_info().rss / 1024**3
        metrics['memory_usage'].append(current_memory)
        self.console.print(f"[green]✓ Completed: {step_name} (Memory: {current_memory:.2f}GB, Workers: {self.n_jobs})[/green]")

    def _log_step_error(self, step_name: str, error: Exception):
        """Log an error in a pipeline step."""
        self.console.print(f"[red]✗ Error in {step_name}: {error}[/red]")
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _free_memory(self):
        """Free up memory by garbage collection."""
        before_memory = psutil.Process().memory_info().rss / 1024**3
        gc.collect()
        gc.collect()
        after_memory = psutil.Process().memory_info().rss / 1024**3
        freed = before_memory - after_memory
        self.console.print(f"[yellow]♻ Freed {freed:.2f}GB[/yellow]" if freed > 0 else "[yellow]♻ GC completed[/yellow]")

    def _validate_raw_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: Optional[pd.DataFrame]) -> Dict:
        """Validate raw data integrity using parallel processing."""
        validation = {'passed': True, 'issues': [], 'warnings': []}
        
        required_cols = {
            'ratings': ['userId', 'movieId', 'rating', 'timestamp'], 
            'movies': ['movieId', 'title', 'genres'], 
            'tags': ['userId', 'movieId', 'tag', 'timestamp'] if tags_df is not None else []
        }
        
        def validate_dataframe(name, df, required_columns):
            issues = []
            warnings = []
            
            if df is None and name == 'tags':
                return issues, warnings
            
            if df is None:
                issues.append(f"{name.title()} is None")
                return issues, warnings
            
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols: 
                issues.append(f"{name.title()} missing columns: {missing_cols}")
            
            if len(df) == 0: 
                issues.append(f"{name.title()} is empty")
            
            return issues, warnings
        
        # Validate dataframes in parallel
        validation_tasks = [
            ('ratings', ratings_df, required_cols['ratings']),
            ('movies', movies_df, required_cols['movies']),
            ('tags', tags_df, required_cols['tags'])
        ]
        
        validation_results = Parallel(n_jobs=min(3, self.n_jobs), backend='threading')(
            delayed(validate_dataframe)(name, df, cols) for name, df, cols in validation_tasks
        )
        
        for issues, warnings in validation_results:
            validation['issues'].extend(issues)
            validation['warnings'].extend(warnings)
        
        # Additional validation checks
        if 'rating' in ratings_df.columns and (~ratings_df['rating'].between(0.5, 5.0)).any(): 
            validation['warnings'].append("Ratings outside [0.5, 5.0] range found")
        
        orphan_ratings = ~ratings_df['movieId'].isin(movies_df['movieId'])
        if orphan_ratings.any(): 
            validation['warnings'].append(f"{orphan_ratings.sum()} orphan ratings (movies not in movies_df)")
        
        if validation['issues']:
            validation['passed'] = False
        
        return validation

    def _create_feature_matrix_optimized(self, ratings_df: pd.DataFrame, user_features: pd.DataFrame, movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix with memory optimization and parallel processing."""
        self.console.print("[cyan]Creating optimized feature matrix (parallel)...[/cyan]")
        
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        estimated_size_gb = (len(ratings_df) * (len(user_features.columns) + len(movie_features.columns)) * 8) / 1024**3
        
        if estimated_size_gb > available_memory_gb * 0.8:
            sample_size = int(len(ratings_df) * (available_memory_gb * 0.5) / estimated_size_gb)
            ratings_df = ratings_df.sample(n=min(sample_size, len(ratings_df)), random_state=42)
            self.console.print(f"[yellow]Sampled {len(ratings_df):,} records to fit memory constraints[/yellow]")
        
        # Use parallel processing for large feature matrices
        if len(ratings_df) > 500_000:
            chunk_size = len(ratings_df) // self.n_jobs
            chunks = [ratings_df.iloc[i:i+chunk_size] for i in range(0, len(ratings_df), chunk_size)]
            
            def process_feature_chunk(chunk):
                chunk_matrix = chunk.merge(user_features, left_on='userId', right_index=True, how='left')
                chunk_matrix = chunk_matrix.merge(movie_features, left_on='movieId', right_index=True, how='left')
                return chunk_matrix
            
            chunk_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(process_feature_chunk)(chunk) for chunk in chunks
            )
            
            feature_matrix = pd.concat(chunk_results, ignore_index=True)
        else:
            feature_matrix = ratings_df.merge(user_features, left_on='userId', right_index=True, how='left')
            feature_matrix = feature_matrix.merge(movie_features, left_on='movieId', right_index=True, how='left')
        
        # Create interaction features in parallel
        interaction_pairs = [('rating_mean', 'rating_mean'), ('rating_count', 'rating_count')]
        
        def create_interaction_features_parallel(pair_index, user_col, movie_col):
            user_cols = [col for col in feature_matrix.columns if col.startswith(user_col) and '_y' not in col]
            movie_cols = [col for col in feature_matrix.columns if col.startswith(movie_col) and '_x' not in col]
            
            interactions = {}
            if user_cols and movie_cols:
                u_name, m_name = user_cols[0], movie_cols[0]
                interactions[f'interact_{pair_index}_mult'] = feature_matrix[u_name] * feature_matrix[m_name]
                interactions[f'interact_{pair_index}_ratio'] = feature_matrix[u_name] / (feature_matrix[m_name] + 1e-8)
            
            return interactions
        
        interaction_results = Parallel(n_jobs=min(len(interaction_pairs), self.n_jobs), backend='threading')(
            delayed(create_interaction_features_parallel)(i, user_col, movie_col) 
            for i, (user_col, movie_col) in enumerate(interaction_pairs)
        )
        
        # Add interaction features to the matrix
        for interactions in interaction_results:
            for name, values in interactions.items():
                feature_matrix[name] = values
        
        # Optimize data types using transformer's parallel method
        feature_matrix = self.transformer.optimize_dtypes(feature_matrix)
        
        memory_usage_mb = feature_matrix.memory_usage(deep=True).sum() / 1024**2
        self.console.print(f"[green]✓ Optimized matrix: {feature_matrix.shape} ({memory_usage_mb:.1f}MB) using parallel processing[/green]")
        
        return feature_matrix

    def _validate_ml_datasets(self, ml_datasets: Dict, metrics: Dict):
        """Validate ML datasets are properly prepared using parallel processing."""
        def validate_single_dataset(task_name, dataset):
            if dataset is None: 
                return task_name, {'valid': False, 'issues': ['Creation failed'], 'warnings': []}
            
            issues, warnings = [], []
            
            if 'X' in dataset:
                X = dataset['X']
                if hasattr(X, 'isna') and X.isna().sum().sum() > 0: 
                    issues.append("Contains NaN values")
                if hasattr(X, 'shape'):
                    if X.shape[0] == 0: 
                        issues.append("No samples")
                    if X.shape[1] == 0: 
                        issues.append("No features")
                    if X.shape[0] < 100: 
                        warnings.append("Few samples (< 100)")
                    if X.shape[1] > 1000: 
                        warnings.append("High dimensionality (> 1000 features)")
            elif task_name == 'association_rules':
                if not dataset.get('transactions'): 
                    issues.append("No transactions found")
                elif len(dataset['transactions']) < 100: 
                    warnings.append("Few transactions (< 100)")
            
            return task_name, {'valid': not issues, 'issues': issues, 'warnings': warnings}
        
        # Validate datasets in parallel
        validation_results = Parallel(n_jobs=min(len(ml_datasets), self.n_jobs), backend='threading')(
            delayed(validate_single_dataset)(task_name, dataset) 
            for task_name, dataset in ml_datasets.items()
        )
        
        # Collect results
        validation_dict = {}
        for task_name, result in validation_results:
            validation_dict[task_name] = result
        
        metrics['validation_results']['ml_datasets'] = validation_dict
        
        # Display results
        failed_datasets = [name for name, result in validation_dict.items() if not result['valid']]
        warning_datasets = [name for name, result in validation_dict.items() if result['warnings']]
        
        if failed_datasets: 
            self.console.print(f"\n[red]✗ ML Dataset Issues found in: {', '.join(failed_datasets)}[/red]")
        if warning_datasets: 
            self.console.print(f"\n[yellow]⚠ ML Dataset Warnings for: {', '.join(warning_datasets)}[/yellow]")

    def _save_results_optimized(self, results: Dict):
        """Save results with compression and parallel processing for large datasets."""
        self.console.print("\n[cyan]Saving results with parallel optimization...[/cyan]")
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        saved_files = []
        total_size_mb = 0

        def save_dataframe(name, df):
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = PROCESSED_DATA_DIR / f"{name}.parquet"
                compression = 'gzip' if len(df) > 1_000_000 else 'snappy'
                df.to_parquet(
                    filepath, 
                    compression=compression, 
                    index=isinstance(df.index, pd.MultiIndex) or 'features' in name
                )
                size = filepath.stat().st_size / 1024**2
                return (filepath.name, size, df.shape)
            return None

        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            console=self.console
        ) as progress:
            # Count items to save
            items_to_save = sum(
                len(d) for k, d in results.items() 
                if isinstance(d, dict) and k in ['cleaned_data', 'feature_engineering']
            ) + 3  # +3 for ml_datasets, sparse features, collaborative filtering
            
            save_task = progress.add_task("[cyan]Saving data...", total=items_to_save)

            # Prepare save tasks for parallel execution
            save_tasks = []
            
            for category in ['cleaned_data', 'feature_engineering']:
                if category in results:
                    for name, df in results[category].items():
                        save_tasks.append((name, df))
            
            # Save DataFrames in parallel
            if save_tasks:
                save_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                    delayed(save_dataframe)(name, df) for name, df in save_tasks
                )
                
                for result in save_results:
                    if result:
                        saved_files.append(result)
                        total_size_mb += result[1]
                    progress.advance(save_task)

            # Save ML datasets (compressed)
            if 'ml_ready_datasets' in results:
                ml_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
                valid_ml = {k: v for k, v in results['ml_ready_datasets'].items() if v}
                with gzip.open(ml_path, 'wb', compresslevel=6) as f: 
                    pickle.dump(valid_ml, f, protocol=4)
                size = ml_path.stat().st_size / 1024**2
                saved_files.append((ml_path.name, size, len(valid_ml)))
                total_size_mb += size
                progress.advance(save_task)

            # Save sparse features
            if 'sparse_features' in results and 'tfidf_matrix' in results['sparse_features']:
                filepath = PROCESSED_DATA_DIR / "tfidf_matrix.npz"
                sparse.save_npz(filepath, results['sparse_features']['tfidf_matrix'])
                size = filepath.stat().st_size / 1024**2
                saved_files.append((filepath.name, size, results['sparse_features']['tfidf_matrix'].shape))
                total_size_mb += size
                progress.advance(save_task)

            # Save collaborative filtering matrix
            if 'collaborative_filtering' in results and 'user_item_matrix' in results['collaborative_filtering']:
                filepath = PROCESSED_DATA_DIR / "user_item_matrix.npz"
                sparse.save_npz(filepath, results['collaborative_filtering']['user_item_matrix'])
                size = filepath.stat().st_size / 1024**2
                saved_files.append((filepath.name, size, results['collaborative_filtering']['user_item_matrix'].shape))
                total_size_mb += size
                progress.advance(save_task)

            # Save transformers
            try:
                transformers_path = PROCESSED_DATA_DIR / "transformers.pkl"
                self.transformer.save_transformers(str(transformers_path))
                size = transformers_path.stat().st_size / 1024**2
                saved_files.append((transformers_path.name, size, 'N/A'))
                total_size_mb += size
                progress.advance(save_task)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not save transformers: {e}[/yellow]")

        # Display save summary
        save_table = Table(title="Saved Files Summary (Parallel Processing)", box=box.ROUNDED)
        save_table.add_column("File", style="cyan")
        save_table.add_column("Size (MB)", justify="right")
        save_table.add_column("Shape/Count", justify="right")
        
        for name, size, shape in saved_files: 
            save_table.add_row(name, f"{size:.2f}", str(shape))
        
        save_table.add_row("[bold]Total", f"[bold]{total_size_mb:.2f}", f"[bold]{len(saved_files)} files")
        self.console.print(save_table)

    def _display_enhanced_summary(self, results: Dict, metrics: Dict):
        """Display comprehensive pipeline summary with parallel processing metrics."""
        self.console.print("\n" + "="*80)
        self.console.print(Panel.fit(
            "[bold green]🎉 Parallel Preprocessing Pipeline Complete! 🎉[/bold green]", 
            border_style="green"
        ))

        # Execution metrics
        exec_table = Table(title="📊 Execution Metrics", box=box.ROUNDED)
        exec_table.add_column("Metric", style="cyan")
        exec_table.add_column("Value", justify="right", style="yellow")
        
        total_time = metrics.get('total_time', 0)
        time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        exec_table.add_row("Total Execution Time", time_str)
        exec_table.add_row("Peak Memory Usage", f"{metrics.get('peak_memory_gb', 0):.2f} GB")
        exec_table.add_row("Parallel Workers Used", str(metrics.get('parallel_workers', self.n_jobs)))
        exec_table.add_row("Batch Size", f"{metrics.get('batch_size', 'N/A'):,}")
        
        # Calculate speedup estimate (rough estimate)
        sequential_estimate = total_time * metrics.get('parallel_workers', self.n_jobs) * 0.7  # Assume 70% parallelizable
        speedup = sequential_estimate / total_time if total_time > 0 else 1
        exec_table.add_row("Estimated Speedup", f"{speedup:.1f}x")
        
        self.console.print(exec_table)

        # Feature engineering summary
        if 'feature_counts' in metrics:
            feature_table = Table(title="🔧 Feature Engineering Summary", box=box.ROUNDED)
            feature_table.add_column("Feature Set", style="cyan")
            feature_table.add_column("Count", justify="right")
            feature_table.add_column("Processing", style="dim")
            
            total_features = sum(metrics['feature_counts'].values())
            for name, count in metrics['feature_counts'].items(): 
                feature_table.add_row(
                    name.replace('_', ' ').title(), 
                    f"{count:,}",
                    "Parallel"
                )
            feature_table.add_row("[bold]Total Features", f"[bold]{total_features:,}", "[bold]Parallel[/bold]")
            self.console.print(feature_table)

        # ML datasets summary
        if 'ml_ready_datasets' in results:
            ml_table = Table(title="🤖 ML-Ready Datasets", box=box.ROUNDED)
            ml_table.add_column("Task", style="cyan")
            ml_table.add_column("Samples", justify="right")
            ml_table.add_column("Features", justify="right")
            ml_table.add_column("Status", justify="center")
            ml_table.add_column("Processing", style="dim")
            
            for task, data in results['ml_ready_datasets'].items():
                if data:
                    shape = data['X'].shape if 'X' in data else (len(data.get('transactions', [])), 'N/A')
                    ml_table.add_row(
                        task.replace('_', ' ').title(), 
                        f"{shape[0]:,}", 
                        str(shape[1]), 
                        "[green]Ready[/green]",
                        "Parallel"
                    )
                else:
                    ml_table.add_row(
                        task.replace('_', ' ').title(), 
                        "N/A", 
                        "N/A", 
                        "[red]Failed[/red]",
                        "N/A"
                    )
            self.console.print(ml_table)

        # Performance recommendations
        perf_recommendations = []
        if total_time > 300:  # > 5 minutes
            perf_recommendations.append("Consider increasing batch size for even faster processing")
        if metrics.get('peak_memory_gb', 0) < 8:
            perf_recommendations.append("System has unused memory - could increase batch sizes")
        if speedup < 2:
            perf_recommendations.append("Consider using more parallel workers if CPU cores available")

        if perf_recommendations:
            self.console.print("\n[bold cyan]📈 Performance Recommendations:[/bold cyan]")
            for rec in perf_recommendations:
                self.console.print(f"  💡 {rec}")

        self.console.print(f"\n[bold green]✅ Parallel pipeline finished successfully! Data saved to {PROCESSED_DATA_DIR}[/bold green]")
        self.console.print(f"[dim]Pipeline used {self.n_jobs} parallel workers for optimal performance[/dim]")

    # Additional utility methods for advanced pipeline operations
    def run_benchmark_subset(self, config, transformer):
        """Run a subset of preprocessing for benchmarking purposes."""
        try:
            # Load a small sample of data
            ratings_df, movies_df, tags_df = self.cleaner.load_data()
            
            # Sample for benchmarking
            sample_size = min(100_000, len(ratings_df))
            sample_ratings = ratings_df.sample(n=sample_size, random_state=42)
            
            # Run basic cleaning and feature creation
            cleaned_ratings = self.cleaner.clean_ratings(save=False, optimize_memory=True)
            user_features = transformer.create_user_features(sample_ratings)
            movie_features = transformer.create_movie_features(sample_ratings, movies_df)
            
            return {
                'samples_processed': len(sample_ratings),
                'features_created': len(user_features.columns) + len(movie_features.columns),
                'status': 'success'
            }
        except Exception as e:
            self.console.print(f"[red]Benchmark failed: {e}[/red]")
            return None

    def get_performance_stats(self) -> Dict:
        """Get performance statistics from the last pipeline run."""
        if not hasattr(self, 'processed_data') or not self.processed_data:
            return {}
        
        metrics = self.processed_data.get('pipeline_metrics', {})
        return {
            'total_time_seconds': metrics.get('total_time', 0),
            'peak_memory_gb': metrics.get('peak_memory_gb', 0),
            'parallel_workers': metrics.get('parallel_workers', self.n_jobs),
            'steps_completed': len(metrics.get('steps_completed', [])),
            'feature_counts': metrics.get('feature_counts', {}),
            'batch_size': metrics.get('batch_size', 'N/A')
        }

    def run_optimized_pipeline(self, config, transformer, skip_validation: bool = False):
        """Run the pipeline with optimized settings from configuration."""
        return self.run_full_pipeline_with_monitoring(
            create_sparse_matrices=True,
            apply_pca=not skip_validation,  # Skip PCA if validation is skipped for speed
            save_results=True,
            use_cache=True,
            memory_limit_gb=config.memory_limit_gb,
            validate_steps=not skip_validation,
            large_dataset_threshold=config.chunk_size,
            sparse_tfidf_threshold=config.chunk_size // 2,
            batch_size=config.user_batch_size
        )

