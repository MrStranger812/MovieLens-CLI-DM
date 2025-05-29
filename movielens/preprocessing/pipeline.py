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

class PreprocessingPipeline:
    """Complete preprocessing pipeline combining cleaning and transformation."""

    def __init__(self):
        self.console = Console()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.processed_data = {} # Will hold loaded/processed data
        self.feature_sets = {}

    def _create_comprehensive_features(self, ratings_df: pd.DataFrame,
                                     user_features: pd.DataFrame,
                                     movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive feature matrix by merging all features."""
        self.console.print("[cyan]Creating comprehensive feature matrix...[/cyan]")
        feature_matrix = ratings_df.copy()
        feature_matrix = feature_matrix.merge(
            user_features, left_on='userId', right_index=True, how='left'
        )
        feature_matrix = feature_matrix.merge(
            movie_features, left_on='movieId', right_index=True, how='left'
        )
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
        self.console.print(f"[green]✓ Comprehensive matrix created: {feature_matrix.shape}[/green]")
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
                    cache_file.unlink() # Remove invalid cache
                    return None
            except Exception as e:
                self.console.print(f"[red]Error loading cache file {cache_file}: {e}. Removing it.[/red]")
                try:
                   cache_file.unlink()
                except OSError:
                   pass # Ignore if already gone
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
                                        large_dataset_threshold: int = 10_000_000,
                                        sparse_tfidf_threshold: int = 1_000_000) -> Optional[Dict]:
        """
        Enhanced preprocessing pipeline with comprehensive monitoring, validation, and memory optimization.
        (Full docstring omitted for brevity - see original file)
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**3  # GB

        self.console.print(Panel.fit(
            "[bold blue]Starting Enhanced Preprocessing Pipeline[/bold blue]\n"
            f"System Memory: {psutil.virtual_memory().percent:.1f}% used "
            f"({psutil.virtual_memory().used / 1024**3:.1f}/{psutil.virtual_memory().total / 1024**3:.1f} GB)\n"
            f"Process Memory: {initial_memory:.2f} GB\n"
            f"Memory Limit: {memory_limit_gb} GB\n"
            f"CPU Count: {psutil.cpu_count()} cores",
            border_style="blue"
        ))

        pipeline_metrics = {
            'start_time': pd.Timestamp.now(), 'steps_completed': [], 'memory_usage': [],
            'validation_results': {}, 'feature_counts': {}, 'data_stats': {}
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
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(),
            TextColumn("| Mem: {task.fields[memory]:.1f}GB"), TextColumn("| {task.fields[status]}"),
            console=self.console, refresh_per_second=2
        ) as progress:
            main_task = progress.add_task("[bold cyan]Pipeline Progress", total=9,
                                        memory=initial_memory, status="Initializing...")
            step_name = "Initialization"
            try:
                # ============ STEP 1: Data Loading ============
                step_name = "Data Loading & Validation"
                self._log_step_start(step_name)
                progress.update(main_task, status="Loading data...")
                ratings_df, movies_df, tags_df = self.cleaner.load_data()
                pipeline_metrics['data_stats']['initial'] = {
                    'ratings_count': len(ratings_df), 'movies_count': len(movies_df),
                    'tags_count': len(tags_df), 'users_count': ratings_df['userId'].nunique(),
                    'memory_usage_mb': sum(df.memory_usage(deep=True).sum() / 1024**2 for df in [ratings_df, movies_df, tags_df] if df is not None)
                }
                if validate_steps:
                    validation_result = self._validate_raw_data(ratings_df, movies_df, tags_df)
                    pipeline_metrics['validation_results']['raw_data'] = validation_result
                    if not validation_result['passed']: raise ValueError(f"Data validation failed: {validation_result['issues']}")
                ratings_df = self.transformer.optimize_dtypes(ratings_df)
                movies_df = self.transformer.optimize_dtypes(movies_df)
                if tags_df is not None: tags_df = self.transformer.optimize_dtypes(tags_df)
                results['raw_data'] = {'ratings': ratings_df, 'movies': movies_df, 'tags': tags_df}
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Data loaded")

                # ============ STEP 2: Data Cleaning ============
                step_name = "Data Cleaning"
                self._log_step_start(step_name)
                progress.update(main_task, status="Cleaning data...")
                cleaning_task = progress.add_task("[cyan]Cleaning datasets...", total=3, memory=process.memory_info().rss / 1024**3, status = "")
                cleaned_ratings = self.cleaner.clean_ratings(save=False, optimize_memory=True)
                progress.advance(cleaning_task)
                cleaned_movies = self.cleaner.clean_movies(save=False)
                progress.advance(cleaning_task)
                cleaned_tags = self.cleaner.clean_tags(save=False) if tags_df is not None else None
                progress.advance(cleaning_task)
                cleaned_ratings = self.transformer.handle_missing_values(cleaned_ratings)
                cleaned_ratings = self.transformer.remove_duplicates(cleaned_ratings, subset=['userId', 'movieId', 'timestamp'])
                cleaned_ratings = self.transformer.detect_outliers(cleaned_ratings, columns=['rating'], method='iqr')
                results['cleaned_data'] = {'ratings': cleaned_ratings, 'movies': cleaned_movies, 'tags': cleaned_tags}
                current_memory = process.memory_info().rss / 1024**3
                if current_memory > memory_limit_gb * 0.8: self._free_memory()
                if current_memory > memory_limit_gb: raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}GB > {memory_limit_gb}GB")
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Data cleaned")

                # ============ STEP 3: Feature Engineering ============
                step_name = "Feature Engineering"
                self._log_step_start(step_name)
                progress.update(main_task, status="Engineering features...")
                feature_task = progress.add_task("[cyan]Creating features...", total=6, memory=process.memory_info().rss / 1024**3, 
                                 status="")
                is_large_dataset = len(cleaned_ratings) > large_dataset_threshold
                if is_large_dataset: self.console.print(f"[yellow]Large dataset detected ({len(cleaned_ratings):,} ratings)[/yellow]")
                
                # 3.1 User features
                progress.update(feature_task, description="[cyan]Creating user features...")
                user_features = self.transformer.create_user_features_optimized(cleaned_ratings, batch_size=50000) if is_large_dataset else self.transformer.create_user_features(cleaned_ratings)
                pipeline_metrics['feature_counts']['user_features'] = len(user_features.columns)
                progress.advance(feature_task)

                # 3.2 Movie features
                progress.update(feature_task, description="[cyan]Creating movie features...")
                movie_features = self.transformer.create_movie_features(cleaned_ratings, cleaned_movies)
                pipeline_metrics['feature_counts']['movie_features'] = len(movie_features.columns)
                progress.advance(feature_task)

                # 3.3 Temporal features *** FIX APPLIED HERE ***
                progress.update(feature_task, description="[cyan]Creating temporal features...")
                temporal_ratings, user_temporal, movie_temporal = self.transformer.create_temporal_features(cleaned_ratings, cleaned_movies) # Pass cleaned_movies
                if not user_temporal.empty: user_features = user_features.join(user_temporal, how='left')
                if not movie_temporal.empty: movie_features = movie_features.join(movie_temporal, how='left')
                pipeline_metrics['feature_counts']['temporal_features'] = (len(user_temporal.columns) + len(movie_temporal.columns))
                progress.advance(feature_task)

                # 3.4 Tag features
                if cleaned_tags is not None and len(cleaned_tags) > 0:
                    progress.update(feature_task, description="[cyan]Creating tag features...")
                    if len(cleaned_tags) > sparse_tfidf_threshold:
                        tfidf_matrix, movie_ids, tfidf = self.transformer.create_sparse_tfidf_features(cleaned_tags, cleaned_movies)
                        results['sparse_features'] = {'tfidf_matrix': tfidf_matrix, 'movie_ids': movie_ids, 'tfidf_vectorizer': tfidf}
                    else:
                        tag_features, tag_transformers = self.transformer.create_tag_features(cleaned_tags, cleaned_movies)
                        movie_features = movie_features.join(tag_features, how='left')
                        results['tag_transformers'] = tag_transformers
                    user_tag_prefs = self.transformer.create_user_tag_preferences(cleaned_ratings, cleaned_tags)
                    user_features = user_features.join(user_tag_prefs, how='left')
                    pipeline_metrics['feature_counts']['tag_features'] = 100
                progress.advance(feature_task)

                # 3.5 Genre features
                progress.update(feature_task, description="[cyan]Creating genre features...")
                genre_features = self.transformer.create_enhanced_genre_features(cleaned_movies, cleaned_ratings)
                # Find overlapping columns between movie_features and genre_features
                overlapping_columns = set(movie_features.columns) & set(genre_features.columns)

                if overlapping_columns:
                    # If there are overlapping columns, specify suffixes
                    movie_features = movie_features.join(genre_features, how='left', lsuffix='_movie', rsuffix='_genre')
                else:
                    # If no overlap, proceed without suffixes
                    movie_features = movie_features.join(genre_features, how='left')
                pipeline_metrics['feature_counts']['genre_features'] = len(genre_features.columns)
                progress.advance(feature_task)

                # 3.6 Cold start features
                progress.update(feature_task, description="[cyan]Creating cold start features...")
                user_cold_features, movie_cold_features = self.transformer.create_cold_start_features(cleaned_ratings, cleaned_movies, user_features, movie_features)
                user_features = user_features.join(user_cold_features, how='left')
                movie_features = movie_features.join(movie_cold_features, how='left')
                pipeline_metrics['feature_counts']['cold_start_features'] = (len(user_cold_features.columns) + len(movie_cold_features.columns))
                progress.advance(feature_task)

                if validate_steps:
                    user_validation = self.transformer.validate_features(user_features, 'user')
                    movie_validation = self.transformer.validate_features(movie_features, 'movie')
                    if user_validation['issues'] or user_validation['warnings']: user_features = self.transformer.auto_fix_features(user_features, user_validation)
                    if movie_validation['issues'] or movie_validation['warnings']: movie_features = self.transformer.auto_fix_features(movie_features, movie_validation)
                    pipeline_metrics['validation_results']['features'] = {'user_features': user_validation, 'movie_features': movie_validation}
                results['feature_engineering'] = {'user_features': user_features, 'movie_features': movie_features, 'feature_matrix': None}
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Features created")

                # ============ STEP 4: Feature Encoding & Normalization ============
                step_name = "Feature Encoding & Normalization"
                self._log_step_start(step_name)
                progress.update(main_task, status="Encoding/Normalizing...")
                sample_size = min(100000, len(temporal_ratings))
                sample_ratings = temporal_ratings.sample(n=sample_size, random_state=42)
                sample_matrix = self._create_comprehensive_features(sample_ratings, user_features, movie_features)
                categorical_columns = ['day_of_week', 'season']
                available_categorical = [col for col in categorical_columns if col in sample_matrix.columns]
                if available_categorical: sample_matrix = self.transformer.encode_categorical_features(sample_matrix, available_categorical, method='label')
                numerical_columns = ['rating', 'year', 'month', 'hour', 'rating_count', 'rating_mean', 'rating_std', 'user_count', 'user_mean', 'user_std', 'movie_count', 'movie_mean', 'movie_std']
                available_numerical = [col for col in numerical_columns if col in sample_matrix.columns]
                if available_numerical: sample_matrix = self.transformer.normalize_features(sample_matrix, available_numerical, method='standard')
                results['encoding_info'] = {'categorical_encoded': available_categorical, 'numerical_normalized': available_numerical, 'sample_shape': sample_matrix.shape}
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Features encoded")

                # ============ STEP 5: Dimensionality Reduction (Optional) ============
                if apply_pca:
                    step_name = "Dimensionality Reduction"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Applying PCA...")
                    user_numeric = user_features.select_dtypes(include=[np.number])
                    movie_numeric = movie_features.select_dtypes(include=[np.number])
                    results['pca_features'] = {}
                    if len(user_numeric.columns) > 50:
                        user_pca, user_pca_summary = self.transformer.apply_pca(user_features, user_numeric.columns.tolist(), variance_threshold=0.95)
                        results['pca_features']['user_pca'] = user_pca
                        results['pca_features']['user_pca_summary'] = user_pca_summary
                    if len(movie_numeric.columns) > 50:
                        movie_pca, movie_pca_summary = self.transformer.apply_pca(movie_features, movie_numeric.columns.tolist(), variance_threshold=0.95)
                        results['pca_features']['movie_pca'] = movie_pca
                        results['pca_features']['movie_pca_summary'] = movie_pca_summary
                    self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="PCA complete" if apply_pca else "Skipped PCA")

                # ============ STEP 6: Create Sparse Matrices ============
                if create_sparse_matrices:
                    step_name = "Creating Sparse Matrices"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Creating sparse matrices...")
                    matrix_cache_file = PROCESSED_DATA_DIR / "user_item_matrix.npz"
                    mappings_cache_file = PROCESSED_DATA_DIR / "user_item_mappings.pkl"
                    if use_cache and matrix_cache_file.exists() and mappings_cache_file.exists():
                        try:
                            user_item_matrix = sparse.load_npz(matrix_cache_file)
                            with open(mappings_cache_file, 'rb') as f: user_mapping, movie_mapping = pickle.load(f)
                            self.console.print("[green]✓ Loaded cached user-item matrix[/green]")
                        except:
                            user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(cleaned_ratings, sparse=True)
                            if save_results: sparse.save_npz(matrix_cache_file, user_item_matrix); pickle.dump((user_mapping, movie_mapping), open(mappings_cache_file, 'wb'))
                    else:
                        user_item_matrix, user_mapping, movie_mapping = self.transformer.create_user_item_matrix(cleaned_ratings, sparse=True)
                        if save_results: sparse.save_npz(matrix_cache_file, user_item_matrix); pickle.dump((user_mapping, movie_mapping), open(mappings_cache_file, 'wb'))
                    results['collaborative_filtering'] = {'user_item_matrix': user_item_matrix, 'user_mapping': user_mapping, 'movie_mapping': movie_mapping}
                    self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="Sparse matrices ready")

                # ============ STEP 7: ML Dataset Preparation ============
                step_name = "ML Dataset Preparation"
                self._log_step_start(step_name)
                progress.update(main_task, status="Preparing ML datasets...")
                ml_task = progress.add_task("[cyan]Preparing ML datasets...", total=5, memory=process.memory_info().rss / 1024**3, 
                             status="")
                ml_datasets = {}
                tasks = ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']
                for task_name in tasks:
                    progress.update(ml_task, description=f"[cyan]Preparing {task_name}...")
                    try:
                        if task_name in ['regression', 'classification']:
                            sample_size_ml = min(5_000_000, len(temporal_ratings)) if is_large_dataset else len(temporal_ratings)
                            sampled_ratings = temporal_ratings.sample(n=sample_size_ml, random_state=42)
                            feature_matrix = self._create_feature_matrix_optimized(sampled_ratings, user_features, movie_features)
                            ml_datasets[task_name] = self._prepare_supervised_dataset(feature_matrix, task_name)
                        elif task_name == 'clustering_users':
                            user_numeric = user_features.select_dtypes(include=[np.number]).fillna(0)
                            ml_datasets[task_name] = {'X': user_numeric, 'feature_names': user_numeric.columns.tolist(), 'user_ids': user_numeric.index}
                        elif task_name == 'clustering_movies':
                            movie_numeric = movie_features.select_dtypes(include=[np.number]).fillna(0)
                            ml_datasets[task_name] = {'X': movie_numeric, 'feature_names': movie_numeric.columns.tolist(), 'movie_ids': movie_numeric.index}
                        elif task_name == 'association_rules':
                            positive_ratings = cleaned_ratings[cleaned_ratings['rating'] >= 4.0]
                            transactions = positive_ratings.groupby('userId')['movieId'].apply(list).tolist()
                            ml_datasets[task_name] = {'transactions': transactions, 'movies_df': cleaned_movies, 'min_support': 0.01, 'min_confidence': 0.5}
                    except Exception as e:
                        self.console.print(f"[red]Error preparing {task_name}: {e}[/red]")
                        ml_datasets[task_name] = None
                    progress.advance(ml_task)
                results['ml_ready_datasets'] = ml_datasets
                if validate_steps: self._validate_ml_datasets(ml_datasets, pipeline_metrics)
                self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="ML datasets ready")

                # ============ STEP 8: Save Results ============
                if save_results:
                    step_name = "Saving Results"
                    self._log_step_start(step_name)
                    progress.update(main_task, status="Saving results...")
                    self._save_results_optimized(results)
                    try:
                        cache_data = {**results, 'pipeline_metrics': pipeline_metrics, 'pipeline_version': '2.0', 'creation_date': pd.Timestamp.now()}
                        with open(cache_file, 'wb') as f: pickle.dump(cache_data, f, protocol=4)
                        self.console.print("[green]✓ Cache saved successfully[/green]")
                    except Exception as e: self.console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
                    self._log_step_complete(step_name, pipeline_metrics, process)
                progress.advance(main_task)
                progress.update(main_task, memory=process.memory_info().rss / 1024**3, status="✓ Complete!", completed=9)

            except Exception as e:
                self._log_step_error(f"Pipeline failed at {step_name}", e)
                return None # Return None on failure

            finally:
                pipeline_metrics['end_time'] = pd.Timestamp.now()
                pipeline_metrics['total_time'] = (pipeline_metrics['end_time'] - pipeline_metrics['start_time']).total_seconds()
                pipeline_metrics['peak_memory_gb'] = max(pipeline_metrics.get('memory_usage', [process.memory_info().rss / 1024**3]))

        self._display_enhanced_summary(results, pipeline_metrics)
        self.processed_data = results
        return results

    def _prepare_supervised_dataset(self, feature_matrix: pd.DataFrame, task_type: str) -> Dict:
        """Prepare dataset for supervised learning tasks."""
        excluded_columns = ['rating', 'userId', 'movieId', 'timestamp']
        # Ensure interaction features are included if they exist
        all_cols = feature_matrix.columns.tolist()
        feature_columns = [col for col in all_cols if col not in excluded_columns and not col.startswith('interact_') and not col.endswith('_scaled') and not col.endswith('_encoded')]
        feature_columns.extend([col for col in all_cols if col.startswith('interact_')])
        feature_columns.extend([col for col in all_cols if col.endswith('_scaled')])
        feature_columns.extend([col for col in all_cols if col.endswith('_encoded')])
        feature_columns = sorted(list(set(feature_columns))) # Unique & sorted

        if not feature_columns: raise ValueError(f"No features available for {task_type}")
        X = feature_matrix[feature_columns].copy()
        for col in X.columns:
            if X[col].dtype.name == 'category': X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
            else: X[col] = X[col].fillna(0)
        y = feature_matrix['rating'] if task_type == 'regression' else (feature_matrix['rating'] >= 4.0).astype(int)
        return {'X': X, 'y': y, 'feature_names': feature_columns, 'n_samples': len(X), 'n_features': len(feature_columns)}

    def _validate_cache(self, cache_data: Dict) -> bool:
        """Validate cached data is still valid."""
        if 'pipeline_version' not in cache_data or cache_data.get('pipeline_version') != '2.0':
            self.console.print("[yellow]Cache version mismatch[/yellow]")
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
        self.console.print(f"[green]✓ Completed: {step_name} (Memory: {current_memory:.2f}GB)[/green]")

    def _log_step_error(self, step_name: str, error: Exception):
        """Log an error in a pipeline step."""
        self.console.print(f"[red]✗ Error in {step_name}: {error}[/red]")
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _free_memory(self):
        """Free up memory by garbage collection."""
        before_memory = psutil.Process().memory_info().rss / 1024**3
        gc.collect(); gc.collect()
        after_memory = psutil.Process().memory_info().rss / 1024**3
        freed = before_memory - after_memory
        self.console.print(f"[yellow]♻ Freed {freed:.2f}GB[/yellow]" if freed > 0 else "[yellow]♻ GC completed[/yellow]")

    def _validate_raw_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: Optional[pd.DataFrame]) -> Dict:
        """Validate raw data integrity."""
        validation = {'passed': True, 'issues': [], 'warnings': []}
        required_cols = {'ratings': ['userId', 'movieId', 'rating', 'timestamp'], 'movies': ['movieId', 'title', 'genres'], 'tags': ['userId', 'movieId', 'tag', 'timestamp'] if tags_df is not None else []}
        for name, df in [('ratings', ratings_df), ('movies', movies_df), ('tags', tags_df)]:
            if df is None and name == 'tags': continue
            missing_cols = set(required_cols[name]) - set(df.columns)
            if missing_cols: validation['passed'] = False; validation['issues'].append(f"{name.title()} missing: {missing_cols}")
            if len(df) == 0: validation['passed'] = False; validation['issues'].append(f"{name.title()} is empty")
        if 'rating' in ratings_df.columns and (~ratings_df['rating'].between(0.5, 5.0)).any(): validation['warnings'].append("Ratings outside [0.5, 5.0]")
        orphan_ratings = ~ratings_df['movieId'].isin(movies_df['movieId'])
        if orphan_ratings.any(): validation['warnings'].append(f"{orphan_ratings.sum()} orphan ratings")
        return validation

    def _create_feature_matrix_optimized(self, ratings_df: pd.DataFrame, user_features: pd.DataFrame, movie_features: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix with memory optimization."""
        self.console.print("[cyan]Creating optimized feature matrix...[/cyan]")
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        estimated_size_gb = (len(ratings_df) * (len(user_features.columns) + len(movie_features.columns)) * 8) / 1024**3
        if estimated_size_gb > available_memory_gb * 0.8:
            sample_size = int(len(ratings_df) * (available_memory_gb * 0.5) / estimated_size_gb)
            ratings_df = ratings_df.sample(n=min(sample_size, len(ratings_df)), random_state=42)
            self.console.print(f"[yellow]Sampled {len(ratings_df):,} records[/yellow]")
        feature_matrix = ratings_df.merge(user_features, left_on='userId', right_index=True, how='left')
        feature_matrix = feature_matrix.merge(movie_features, left_on='movieId', right_index=True, how='left')
        interaction_pairs = [('rating_mean', 'rating_mean'), ('rating_count', 'rating_count')]
        for i, (user_col, movie_col) in enumerate(interaction_pairs):
            user_cols = [col for col in feature_matrix.columns if col.startswith(user_col) and '_y' not in col]
            movie_cols = [col for col in feature_matrix.columns if col.startswith(movie_col) and '_x' not in col]
            if user_cols and movie_cols:
                u_name, m_name = user_cols[0], movie_cols[0]
                feature_matrix[f'interact_{i}_mult'] = feature_matrix[u_name] * feature_matrix[m_name]
                feature_matrix[f'interact_{i}_ratio'] = feature_matrix[u_name] / (feature_matrix[m_name] + 1e-8)
        feature_matrix = self.transformer.optimize_dtypes(feature_matrix)
        self.console.print(f"[green]✓ Optimized matrix: {feature_matrix.shape} ({feature_matrix.memory_usage(deep=True).sum() / 1024**2:.1f}MB)[/green]")
        return feature_matrix

    def _validate_ml_datasets(self, ml_datasets: Dict, metrics: Dict):
        """Validate ML datasets are properly prepared."""
        validation_results = {}
        for task_name, dataset in ml_datasets.items():
            if dataset is None: validation_results[task_name] = {'valid': False, 'issues': ['Creation failed']}; continue
            issues, warnings = [], []
            if 'X' in dataset:
                X = dataset['X']
                if hasattr(X, 'isna') and X.isna().sum().sum() > 0: issues.append("Contains NaNs")
                if hasattr(X, 'shape'):
                    if X.shape[0] == 0: issues.append("No samples")
                    if X.shape[1] == 0: issues.append("No features")
                    if X.shape[0] < 100: warnings.append("Few samples")
                    if X.shape[1] > 1000: warnings.append("High dimensionality")
            elif task_name == 'association_rules':
                if not dataset.get('transactions'): issues.append("No transactions")
                elif len(dataset['transactions']) < 100: warnings.append("Few transactions")
            validation_results[task_name] = {'valid': not issues, 'issues': issues, 'warnings': warnings}
        metrics['validation_results']['ml_datasets'] = validation_results
        if any(not v['valid'] for v in validation_results.values()): self.console.print("\n[red]✗ ML Dataset Issues[/red]")
        if any(v['warnings'] for v in validation_results.values()): self.console.print("\n[yellow]⚠ ML Dataset Warnings[/yellow]")

    def _save_results_optimized(self, results: Dict):
        """Save results with compression and chunking for large datasets."""
        self.console.print("\n[cyan]Saving results with optimization...[/cyan]")
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        saved_files = []
        total_size_mb = 0

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=self.console) as progress:
            items_to_save = sum(len(d) for k, d in results.items() if isinstance(d, dict) and k in ['cleaned_data', 'feature_engineering']) + 3 # +3 for ml, sparse, cf
            save_task = progress.add_task("[cyan]Saving data...", total=items_to_save)

            for category in ['cleaned_data', 'feature_engineering']:
                if category in results:
                    for name, df in results[category].items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            filepath = PROCESSED_DATA_DIR / f"{name}.parquet"
                            df.to_parquet(filepath, compression='gzip' if len(df) > 1_000_000 else 'snappy', index=isinstance(df.index, pd.MultiIndex) or name != 'cleaned_data')
                            size = filepath.stat().st_size / 1024**2
                            saved_files.append((filepath.name, size, df.shape))
                            total_size_mb += size
                            progress.advance(save_task)

            if 'ml_ready_datasets' in results:
                ml_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
                valid_ml = {k: v for k, v in results['ml_ready_datasets'].items() if v}
                with gzip.open(ml_path, 'wb', compresslevel=6) as f: pickle.dump(valid_ml, f, protocol=4)
                size = ml_path.stat().st_size / 1024**2; saved_files.append((ml_path.name, size, len(valid_ml))); total_size_mb += size; progress.advance(save_task)

            if 'sparse_features' in results and 'tfidf_matrix' in results['sparse_features']:
                filepath = PROCESSED_DATA_DIR / "tfidf_matrix.npz"
                sparse.save_npz(filepath, results['sparse_features']['tfidf_matrix'])
                size = filepath.stat().st_size / 1024**2; saved_files.append((filepath.name, size, results['sparse_features']['tfidf_matrix'].shape)); total_size_mb += size; progress.advance(save_task)

            if 'collaborative_filtering' in results and 'user_item_matrix' in results['collaborative_filtering']:
                filepath = PROCESSED_DATA_DIR / "user_item_matrix.npz"
                sparse.save_npz(filepath, results['collaborative_filtering']['user_item_matrix'])
                size = filepath.stat().st_size / 1024**2; saved_files.append((filepath.name, size, results['collaborative_filtering']['user_item_matrix'].shape)); total_size_mb += size; progress.advance(save_task)

            transformers_path = PROCESSED_DATA_DIR / "transformers.pkl"
            self.transformer.save_transformers(transformers_path)
            size = transformers_path.stat().st_size / 1024**2; saved_files.append((transformers_path.name, size, 'N/A')); total_size_mb += size; progress.advance(save_task)

        save_table = Table(title="Saved Files Summary", box=box.ROUNDED)
        save_table.add_column("File", style="cyan"); save_table.add_column("Size (MB)", justify="right"); save_table.add_column("Shape/Count", justify="right")
        for name, size, shape in saved_files: save_table.add_row(name, f"{size:.2f}", str(shape))
        save_table.add_row("[bold]Total", f"[bold]{total_size_mb:.2f}", f"[bold]{len(saved_files)} files")
        self.console.print(save_table)

    def _display_enhanced_summary(self, results: Dict, metrics: Dict):
        """Display comprehensive pipeline summary with metrics."""
        self.console.print("\n" + "="*70)
        self.console.print(Panel.fit("[bold green]🎉 Pipeline Execution Complete! 🎉[/bold green]", border_style="green"))

        exec_table = Table(title="📊 Execution Metrics", box=box.ROUNDED)
        exec_table.add_column("Metric", style="cyan"); exec_table.add_column("Value", justify="right", style="yellow")
        total_time = metrics.get('total_time', 0); time_str = f"{int(total_time // 60)}m {int(total_time % 60)}s"
        exec_table.add_row("Total Execution Time", time_str)
        exec_table.add_row("Peak Memory Usage", f"{metrics.get('peak_memory_gb', 0):.2f} GB")
        self.console.print(exec_table)

        if 'feature_counts' in metrics:
            feature_table = Table(title="🔧 Feature Engineering Summary", box=box.ROUNDED)
            feature_table.add_column("Feature Set", style="cyan"); feature_table.add_column("Count", justify="right")
            total_features = sum(metrics['feature_counts'].values())
            for name, count in metrics['feature_counts'].items(): feature_table.add_row(name.replace('_', ' ').title(), f"{count:,}")
            feature_table.add_row("[bold]Total Features", f"[bold]{total_features:,}")
            self.console.print(feature_table)

        if 'ml_ready_datasets' in results:
            ml_table = Table(title="🤖 ML-Ready Datasets", box=box.ROUNDED)
            ml_table.add_column("Task", style="cyan"); ml_table.add_column("Samples", justify="right"); ml_table.add_column("Features", justify="right"); ml_table.add_column("Status", justify="center")
            for task, data in results['ml_ready_datasets'].items():
                if data:
                    shape = data['X'].shape if 'X' in data else (len(data.get('transactions', [])), 'N/A')
                    ml_table.add_row(task.replace('_', ' ').title(), f"{shape[0]:,}", str(shape[1]), "[green]Ready[/green]")
                else:
                    ml_table.add_row(task.replace('_', ' ').title(), "N/A", "N/A", "[red]Failed[/red]")
            self.console.print(ml_table)

        self.console.print(f"\n[bold green]✅ Pipeline finished. Data saved to {PROCESSED_DATA_DIR}[/bold green]")