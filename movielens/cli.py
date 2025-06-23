import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from .config import PROCESSED_DATA_DIR, REPORTS_DIR
import pickle
import gzip
import multiprocessing as mp
import psutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import subprocess
import platform
import json
from typing import Dict, List, Optional

console = Console()

class EnvironmentManager:
    """Manages conda environments and their configurations."""
    
    def __init__(self):
        self.environments = {
            'base': {
                'name': 'movielens-base',
                'description': 'Base environment for common data processing',
                'requirements': 'requirements/base.txt'
            },
            'gpu': {
                'name': 'movielens-rapids',
                'description': 'GPU-accelerated environment for deep learning',
                'requirements': 'requirements/gpu.txt'
            },
            'profiling': {
                'name': 'movielens-profiling',
                'description': 'Environment for performance profiling',
                'requirements': 'requirements/profiling.txt'
            },
            'optimization': {
                'name': 'movielens-optimization',
                'description': 'Environment for performance optimization',
                'requirements': 'requirements/optimization.txt'
            },
            'dev': {
                'name': 'movielens-dev',
                'description': 'Development environment with all tools',
                'requirements': ['requirements/base.txt', 'requirements/gpu.txt', 
                               'requirements/profiling.txt', 'requirements/optimization.txt']
            }
        }
    
    def get_current_environment(self) -> str:
        """Get the current conda environment name."""
        try:
            result = subprocess.run(['conda', 'info', '--envs'], 
                                 capture_output=True, text=True)
            active_env = [line for line in result.stdout.split('\n') 
                         if '*' in line][0].split()[0]
            return active_env
        except:
            return "base"
    
    def check_environment_compatibility(self, env_name: str, operation: str) -> bool:
        """Check if the current environment is compatible with the operation."""
        current_env = self.get_current_environment()
        env_info = self.environments.get(env_name, {})
        
        if not env_info:
            console.print(f"[red]❌ Environment {env_name} not found[/red]")
            return False
        
        if current_env != env_info['name']:
            console.print(f"[yellow]⚠ Warning: Operation {operation} is recommended in {env_info['name']} environment[/yellow]")
            if not Confirm.ask(f"Continue with current environment ({current_env})?", default=False):
                return False
        
        return True

# Initialize environment manager
env_manager = EnvironmentManager()

# Lazy imports for preprocessing modules to avoid dependency issues
def get_data_cleaner():
    """Lazy import of DataCleaner to avoid preprocessing dependencies."""
    from .preprocessing.cleaner import DataCleaner
    return DataCleaner

def get_preprocessing_pipeline():
    """Lazy import of PreprocessingPipeline."""
    from .preprocessing.pipeline import PreprocessingPipeline
    return PreprocessingPipeline

def get_hyper_optimized_transformer():
    """Lazy import of HyperOptimizedDataTransformer."""
    from .preprocessing.transformer_ultrafast import HyperOptimizedDataTransformer
    return HyperOptimizedDataTransformer

class PerformanceConfig:
    """Configuration for performance optimization."""
    
    def __init__(self, 
                 mode: str = 'balanced',
                 n_jobs: int = -1,
                 use_dask: bool = False,
                 memory_limit_gb: float = 16.0):
        
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.use_dask = use_dask
        self.memory_limit_gb = memory_limit_gb
        
        if mode == 'speed':
            self.user_batch_size = 300_000
            self.movie_batch_size = 150_000
            self.chunk_size = 2_000_000
            self.max_tfidf_features = 200
            self.skip_validation = True
            self.enable_numba = True
            
        elif mode == 'memory':
            self.user_batch_size = 50_000
            self.movie_batch_size = 25_000
            self.chunk_size = 200_000
            self.max_tfidf_features = 50
            self.skip_validation = False
            self.enable_numba = False
            
        else:  # balanced
            self.user_batch_size = 150_000
            self.movie_batch_size = 75_000
            self.chunk_size = 1_000_000
            self.max_tfidf_features = 100
            self.skip_validation = False
            self.enable_numba = True
    
    def get_transformer(self):
        """Get configured transformer instance."""
        HyperOptimizedDataTransformer = get_hyper_optimized_transformer()
        return HyperOptimizedDataTransformer(n_jobs=self.n_jobs)

@click.group()
def cli():
    """MovieLens Multi-Analytics CLI - Enhanced Performance Edition"""
    console.print(Panel.fit(
        "[bold blue]MovieLens Multi-Analytics Project[/bold blue]\n"
        "Comprehensive data science analysis tool with hyper-optimized processing\n"
        f"System: {mp.cpu_count()} cores, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM\n"
        f"Current Environment: {env_manager.get_current_environment()}",
        border_style="blue"
    ))

@cli.group()
def env():
    """Environment management commands."""
    pass

@env.command('list')
def list_environments():
    """List available conda environments."""
    table = Table(title="Available Environments", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", style="yellow")
    
    current_env = env_manager.get_current_environment()
    
    for env_name, env_info in env_manager.environments.items():
        status = "✓ Active" if env_info['name'] == current_env else "Inactive"
        table.add_row(env_info['name'], env_info['description'], status)
    
    console.print(table)

@env.command('activate')
@click.argument('env_name')
def activate_environment(env_name):
    """Activate a specific conda environment."""
    if env_name not in env_manager.environments:
        console.print(f"[red]❌ Environment {env_name} not found[/red]")
        return
    
    env_info = env_manager.environments[env_name]
    try:
        if platform.system() == "Windows":
            activate_cmd = f"conda activate {env_info['name']}"
        else:
            activate_cmd = f"source activate {env_info['name']}"
        
        subprocess.run(activate_cmd, shell=True, check=True)
        console.print(f"[green]✓ Activated environment: {env_info['name']}[/green]")
    except subprocess.CalledProcessError:
        console.print(f"[red]❌ Failed to activate environment: {env_info['name']}[/red]")

@cli.command()
@click.option('--technique', type=click.Choice(['regression', 'classification', 'clustering', 'recommender', 'association', 'all']),
              default='all', help='Analysis technique to run')
def analyze(technique):
    """Run data analysis with specified technique."""
    if not env_manager.check_environment_compatibility('base', 'analysis'):
        return
    
    console.print(f"[yellow]Placeholder: Starting {technique} analysis...[/yellow]")
    console.print("[yellow]Note: Analysis functions need to be implemented.[/yellow]")

@cli.command()
def explore():
    """Explore the basic dataset info."""
    if not env_manager.check_environment_compatibility('base', 'exploration'):
        return
    
    console.print("[yellow]Loading basic dataset info...[/yellow]")
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    cleaner.load_data()
    cleaner.basic_info()

@cli.command()
@click.option('--skip-pca', is_flag=True, help='Skip PCA dimensionality reduction')
@click.option('--skip-sparse', is_flag=True, help='Skip sparse matrix creation')
@click.option('--no-save', is_flag=True, help='Do not save processed data')
@click.option('--no-cache', is_flag=True, help='Do not use cached results')
@click.option('--clear-cache', is_flag=True, help='Clear existing cache and run fresh')
@click.option('--memory-limit', type=float, default=16.0, help='Memory limit in GB')
@click.option('--no-validation', is_flag=True, help='Skip validation steps')
@click.option('--batch-size', type=int, default=200000, help='Batch size for processing')
@click.option('--n-jobs', type=int, default=4, help='Number of parallel jobs')
@click.option('--use-gpu', is_flag=True, help='Use GPU acceleration if available')
def preprocess(skip_pca, skip_sparse, no_save, no_cache, clear_cache, memory_limit, 
               no_validation, batch_size, n_jobs, use_gpu):
    """Run hyper-optimized preprocessing pipeline with performance tuning."""
    
    # Check environment compatibility
    env_name = 'gpu' if use_gpu else 'optimization'
    if not env_manager.check_environment_compatibility(env_name, 'preprocessing'):
        return
    
    # Import preprocessing modules only when needed
    from .preprocessing.cleaner import DataCleaner
    from .preprocessing.pipeline import PreprocessingPipeline
    from .preprocessing.transformer import DataTransformer
    from .preprocessing.transformer_ultrafast import HyperOptimizedDataTransformer
    
    # Display system information
    console.print("\n[bold cyan]System Information[/bold cyan]")
    system_table = Table(box=box.ROUNDED)
    system_table.add_column("Resource", style="cyan")
    system_table.add_column("Available", style="green")
    system_table.add_column("Configuration", style="yellow")
    
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024**3
    actual_n_jobs = n_jobs if n_jobs > 0 else cpu_count
    
    system_table.add_row("CPU Cores", str(cpu_count), f"Using {actual_n_jobs} cores")
    system_table.add_row("Memory", f"{memory_gb:.1f} GB", f"Limit: {memory_limit} GB")
    system_table.add_row("Batch Size", f"{batch_size:,}", "Hyper-optimized processing")
    
    console.print(system_table)
    
    # Handle cache clearing
    if clear_cache:
        console.print("[yellow]Clearing cache files...[/yellow]")
        files_to_clear = [
            "preprocessing_cache.pkl",
            "user_item_matrix.npz",
            "user_item_mappings.pkl",
            "ml_ready_datasets.pkl.gz",
            "tfidf_matrix.npz",
            "tfidf_vectorizer.pkl",
            "transformers.pkl"
        ]
        
        for filename in files_to_clear:
            cache_path = PROCESSED_DATA_DIR / filename
            if cache_path.exists():
                cache_path.unlink()
                console.print(f"[yellow]  ✓ Cleared: {cache_path.name}[/yellow]")

    if not Confirm.ask("Proceed with hyper-optimized preprocessing?", default=True):
        console.print("[yellow]Preprocessing cancelled.[/yellow]")
        return
    
    # Start timing
    start_time = time.time()
    
    try:
        console.print("\n[bold green]Starting Hyper-Optimized Preprocessing Pipeline[/bold green]")
        
        # Initialize data cleaner
        cleaner = DataCleaner()
        ratings_df, movies_df, tags_df = cleaner.load_data()
        
        # Clean data
        console.print("[cyan]Cleaning data...[/cyan]")
        cleaned_ratings = cleaner.clean_ratings(save=False)
        cleaned_movies = cleaner.clean_movies(save=False)
        cleaned_tags = cleaner.clean_tags(save=False) if tags_df is not None else None
        
        # Initialize hyper-optimized transformer
        transformer = HyperOptimizedDataTransformer(n_jobs=actual_n_jobs)
        
        # Run feature engineering
        console.print("[cyan]Running hyper-optimized feature engineering...[/cyan]")
        features = transformer.create_all_features_pipeline_gpu(
            cleaned_ratings, cleaned_movies, cleaned_tags
        )
        
        # Create ML datasets
        console.print("[cyan]Creating ML-ready datasets...[/cyan]")
        ml_datasets = create_ml_datasets_hyper_optimized(
            cleaned_ratings, features, cleaned_movies, actual_n_jobs
        )
        
        # Save results
        if not no_save:
            console.print("[cyan]Saving results...[/cyan]")
            save_results_hyper_optimized(features, ml_datasets, actual_n_jobs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        console.print(f"\n[bold green]✅ Hyper-optimized pipeline completed in {total_time:.2f} seconds![/bold green]")
        
        # Display performance metrics
        display_hyper_performance_metrics(features, ml_datasets, total_time, actual_n_jobs)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

@cli.command()
@click.option('--performance-mode', type=click.Choice(['speed', 'balanced', 'memory']), 
              default='balanced', help='Performance optimization mode')
@click.option('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
@click.option('--batch-size', type=int, help='Batch size for processing (auto if not specified)')
@click.option('--memory-limit', type=float, help='Memory limit in GB (auto if not specified)')
@click.option('--skip-validation', is_flag=True, help='Skip validation steps for speed')
@click.option('--profile', is_flag=True, help='Enable performance profiling')
@click.option('--use-gpu', is_flag=True, help='Use GPU acceleration if available')
def preprocess_fast(performance_mode, n_jobs, batch_size, memory_limit, 
                   skip_validation, profile, use_gpu):
    """Run hyper-optimized preprocessing pipeline with performance tuning."""
    
    # Check environment compatibility
    env_name = 'gpu' if use_gpu else 'profiling' if profile else 'optimization'
    if not env_manager.check_environment_compatibility(env_name, 'fast preprocessing'):
        return
    
    # Import preprocessing modules only when needed
    from .preprocessing.cleaner import DataCleaner
    from .preprocessing.transformer_ultrafast import HyperOptimizedDataTransformer
    
    # Display system information
    console.print("\n[bold cyan]System Information[/bold cyan]")
    system_table = Table(box=box.ROUNDED)
    system_table.add_column("Resource", style="cyan")
    system_table.add_column("Available", style="green")
    system_table.add_column("Configuration", style="yellow")
    
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024**3
    actual_n_jobs = n_jobs if n_jobs != -1 else cpu_count
    
    system_table.add_row("CPU Cores", str(cpu_count), f"Using {actual_n_jobs} cores")
    system_table.add_row("Memory", f"{memory_gb:.1f} GB", f"Limit: {memory_limit or 'Auto'} GB")
    system_table.add_row("Mode", performance_mode.title(), "Hyper-optimized processing")
    
    console.print(system_table)
    
    # Create performance configuration
    config = PerformanceConfig(
        mode=performance_mode,
        n_jobs=n_jobs,
        use_dask=False,
        memory_limit_gb=memory_limit or (memory_gb * 0.8)
    )
    
    # Override batch size if specified
    if batch_size:
        config.user_batch_size = batch_size
        config.movie_batch_size = batch_size // 2
    
    # Display configuration
    console.print("\n[bold cyan]Hyper-Optimization Configuration[/bold cyan]")
    config_table = Table(box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    
    config_table.add_row("User Batch Size", f"{config.user_batch_size:,}")
    config_table.add_row("Movie Batch Size", f"{config.movie_batch_size:,}")
    config_table.add_row("Chunk Size", f"{config.chunk_size:,}")
    config_table.add_row("Skip Validation", str(config.skip_validation))
    config_table.add_row("Workers", str(config.n_jobs))
    config_table.add_row("Numba Enabled", str(config.enable_numba))
    
    console.print(config_table)
    
    if not Confirm.ask("Proceed with these hyper-optimized settings?", default=True):
        console.print("[yellow]Preprocessing cancelled.[/yellow]")
        return
    
    # Start timing
    start_time = time.time()
    
    # Enable profiling if requested
    if profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        console.print("\n[bold green]Starting Hyper-Optimized Fast Preprocessing[/bold green]")
        
        # Initialize components
        cleaner = DataCleaner()
        transformer = config.get_transformer()
        
        # Load and clean data
        ratings_df, movies_df, tags_df = cleaner.load_data()
        cleaned_ratings = cleaner.clean_ratings(save=False)
        cleaned_movies = cleaner.clean_movies(save=False)
        cleaned_tags = cleaner.clean_tags(save=False) if tags_df is not None else None
        
        # Run hyper-optimized feature engineering
        features = transformer.create_all_features_pipeline_hyper(
            cleaned_ratings, cleaned_movies, cleaned_tags
        )
        
        # Create ML datasets
        ml_datasets = create_ml_datasets_hyper_optimized(
            cleaned_ratings, features, cleaned_movies, config.n_jobs
        )
        
        # Save results
        save_results_hyper_optimized(features, ml_datasets, config.n_jobs)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        console.print(f"\n[bold green]✅ Hyper-optimized fast preprocessing completed in {total_time:.2f} seconds![/bold green]")
        
        # Display performance metrics
        display_hyper_performance_metrics(features, ml_datasets, total_time, config.n_jobs)
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        
    finally:
        # Save profiling results
        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            console.print("\n[bold cyan]Performance Profile (Top 10 Functions)[/bold cyan]")
            stats.print_stats(10)

@cli.command()
@click.option('--iterations', type=int, default=3, help='Number of benchmark iterations')
@click.option('--modes', type=str, default='speed,balanced,memory', help='Modes to benchmark (comma-separated)')
def benchmark(iterations, modes):
    """Run preprocessing benchmarks to compare performance modes."""
    
    # Import preprocessing modules only when needed
    from .preprocessing.cleaner import DataCleaner
    
    console.print("[bold cyan]Starting Hyper-Optimized Preprocessing Benchmarks[/bold cyan]")
    
    modes_list = [mode.strip() for mode in modes.split(',')]
    results = {}
    
    for mode in modes_list:
        console.print(f"\n[yellow]Benchmarking {mode} mode with hyper-optimization...[/yellow]")
        
        mode_results = []
        
        for i in range(iterations):
            console.print(f"[dim]Iteration {i+1}/{iterations}[/dim]")
            
            # Create configuration
            config = PerformanceConfig(mode=mode)
            
            # Time the preprocessing
            start_time = time.time()
            
            try:
                # Initialize components
                cleaner = DataCleaner()
                transformer = config.get_transformer()
                
                # Run benchmark subset
                ratings_df, movies_df, tags_df = cleaner.load_data()
                
                # Sample data for benchmarking
                sample_size = min(100000, len(ratings_df))
                ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
                
                # Run hyper-optimized processing on sample
                features = transformer.create_all_features_pipeline_hyper(
                    ratings_sample, movies_df, tags_df
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                mode_results.append({
                    'time': elapsed_time,
                    'memory_peak': psutil.Process().memory_info().rss / 1024**3,
                    'success': True
                })
                    
            except Exception as e:
                console.print(f"[red]Error in iteration {i+1}: {e}[/red]")
                mode_results.append({
                    'time': float('inf'),
                    'memory_peak': 0,
                    'success': False
                })
        
        results[mode] = mode_results
    
    # Display benchmark results
    display_benchmark_results(results)

@cli.command()
def summary():
    """Show preprocessing summary statistics from cache."""
    
    cache_file = PROCESSED_DATA_DIR / "hyper_features.pkl.gz"
    ml_datasets_gz_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"

    if cache_file.exists():
        console.print(f"[yellow]Loading hyper-optimized cache: {cache_file}...[/yellow]")
        try:
            with gzip.open(cache_file, 'rb') as f:
                cached_features = pickle.load(f)

            console.print("[green]✓ Loaded hyper-optimized features. Displaying summary...[/green]")
            display_features_summary(cached_features)

        except Exception as e:
            console.print(f"[red]Error loading cache: {e}[/red]")

    elif ml_datasets_gz_path.exists():
        console.print(f"[yellow]Loading ML datasets from {ml_datasets_gz_path}...[/yellow]")
        try:
            with gzip.open(ml_datasets_gz_path, 'rb') as f:
                ml_datasets = pickle.load(f)

            console.print("[green]✓ Found processed ML datasets:[/green]")
            table = Table(title="Available ML Datasets", box=box.ROUNDED)
            table.add_column("Task", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            for task, data in ml_datasets.items():
                if data:
                    shape_info = f"{data['X'].shape}" if 'X' in data and hasattr(data['X'], 'shape') else f"{len(data.get('transactions', []))} transactions"
                    table.add_row(task.replace('_', ' ').title(), "✓ Ready", shape_info)
                else:
                    table.add_row(task.replace('_', ' ').title(), "✗ Failed", "No data")
            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading ML datasets: {e}[/red]")
    else:
        console.print("[yellow]No processed data or cache found.[/yellow]")
        console.print(Panel(
            "[yellow]💡 Tip: Run 'python analyze.py preprocess' to create hyper-optimized datasets[/yellow]",
            border_style="yellow"
        ))

@cli.command('get-dataset')
@click.argument('task', type=click.Choice(['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']))
def get_dataset(task):
    """Get information about a specific preprocessed dataset."""
    
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    
    if not ml_datasets_path.exists():
        console.print(f"[red]Dataset for {task} not found. Run 'preprocess' first.[/red]")
        return
    
    try:
        with gzip.open(ml_datasets_path, 'rb') as f:
            ml_datasets = pickle.load(f)
        
        dataset = ml_datasets.get(task)
        
        if not dataset:
            console.print(f"[red]Dataset for {task} not found in processed data.[/red]")
            return

        console.print(f"\n[bold cyan]Dataset Information: {task.replace('_', ' ').title()}[/bold cyan]")

        info_table = Table(box=box.ROUNDED, title="Dataset Details")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        if 'X' in dataset and hasattr(dataset['X'], 'shape'):
            samples, features = dataset['X'].shape
            info_table.add_row("Samples", f"{samples:,}")
            info_table.add_row("Features", f"{features}")

            if 'y' in dataset and hasattr(dataset['y'], 'shape'):
                info_table.add_row("Target Samples", f"{dataset['y'].shape[0]:,}")
                if hasattr(dataset['y'], 'nunique'):
                    info_table.add_row("Unique Targets", f"{dataset['y'].nunique()}")

            if 'feature_names' in dataset:
                feature_sample = dataset['feature_names'][:5]
                if len(dataset['feature_names']) > 5:
                    feature_sample.append(f"... {len(dataset['feature_names']) - 5} more")
                info_table.add_row("Sample Features", ", ".join(map(str, feature_sample)))

        elif task == 'association_rules':
            transactions = dataset.get('transactions', [])
            info_table.add_row("Type", "Association Rules")
            info_table.add_row("Transactions", f"{len(transactions):,}")
            info_table.add_row("Min Support", str(dataset.get('min_support')))
            info_table.add_row("Min Confidence", str(dataset.get('min_confidence')))

        console.print(info_table)
        
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")

@cli.command()
def validate():
    """Validate all preprocessed ML datasets."""
    
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    all_tasks = ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']
    validation_table = Table(title="Data Validation Results", box=box.ROUNDED)
    validation_table.add_column("ML Task", style="cyan")
    validation_table.add_column("Status", style="bold")
    validation_table.add_column("Details", style="dim")

    if not ml_datasets_path.exists():
        for task_name in all_tasks:
            validation_table.add_row(
                task_name.replace('_', ' ').title(),
                "[red]✗ Missing[/red]",
                "No processed data found"
            )
        console.print(validation_table)
        console.print(Panel(
            "[dim]💡 No datasets found. Run:[/dim]\n"
            "[bold cyan]python analyze.py preprocess[/bold cyan]",
            border_style="blue"
        ))
        return
    
    try:
        with gzip.open(ml_datasets_path, 'rb') as f:
            ml_datasets = pickle.load(f)
        
        found_any = False
        for task_name in all_tasks:
            dataset = ml_datasets.get(task_name)

            if not dataset:
                validation_table.add_row(
                    task_name.replace('_', ' ').title(),
                    "[red]✗ Missing[/red]",
                    "Dataset not found"
                )
            else:
                found_any = True
                if 'X' in dataset and hasattr(dataset['X'], 'shape'):
                    samples, features = dataset['X'].shape
                    nan_count = dataset['X'].isna().sum().sum() if hasattr(dataset['X'], 'isna') else 0
                    status = "[green]✓ Ready[/green]" if nan_count == 0 else "[yellow]⚠ NaN Found[/yellow]"
                    details = f"{samples:,} samples, {features} features"
                    if nan_count > 0:
                        details += f" ({nan_count} NaNs)"
                    validation_table.add_row(task_name.replace('_', ' ').title(), status, details)

                elif task_name == 'association_rules' and 'transactions' in dataset:
                    transaction_count = len(dataset['transactions'])
                    status = "[green]✓ Ready[/green]" if transaction_count > 0 else "[yellow]⚠ Empty[/yellow]"
                    validation_table.add_row(task_name.replace('_', ' ').title(), status, f"{transaction_count:,} transactions")
                else:
                    validation_table.add_row(task_name.replace('_', ' ').title(), "[yellow]⚠ Incomplete[/yellow]", "Unexpected format")

        console.print(validation_table)

        if not found_any:
            console.print(Panel(
                "[dim]💡 No datasets found. Run:[/dim]\n"
                "[bold cyan]python analyze.py preprocess[/bold cyan]",
                border_style="blue"
            ))
            
    except Exception as e:
        console.print(f"[red]Error validating datasets: {e}[/red]")

@cli.command() 
def clean():
    """Run basic data cleaning (demonstration)."""
    DataCleaner = get_data_cleaner()
    console.print("[yellow]Running basic data cleaning...[/yellow]")
    cleaner = DataCleaner()
    cleaner.load_data()
    cleaner.clean_ratings(save=False)
    cleaner.clean_movies(save=False)
    cleaner.clean_tags(save=False)
    console.print("[green]✓ Basic data cleaning demonstration completed (data not saved).[/green]")

# Models Commands

@cli.command()
@click.option('--min-support', type=float, default=0.01, help='Minimum support threshold')
@click.option('--min-confidence', type=float, default=0.5, help='Minimum confidence threshold')
@click.option('--top-rules', type=int, default=10, help='Number of top rules to display')
def association(min_support, min_confidence, top_rules):
    """Run association rule mining on movie viewing patterns."""
    from .models.association import run_association_mining_pipeline
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]Association Rule Mining[/bold cyan]\n"
        "Discovering patterns in movie viewing behaviors",
        border_style="cyan"
    ))
    
    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, _ = cleaner.load_data()
    
    # Run association mining
    miner = run_association_mining_pipeline(
        ratings_df, movies_df, 
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    # Show top rules
    console.print(f"\n[bold]Top {top_rules} Association Rules:[/bold]")
    top_rules_df = miner.get_top_rules(n=top_rules)
    console.print(top_rules_df)
    
    # Show movie bundles
    console.print("\n[bold]Frequent Movie Bundles:[/bold]")
    bundles = miner.get_movie_bundles(n=10)
    console.print(bundles)


@cli.command()
@click.option('--method', type=click.Choice(['all', 'batch', 'sgd', 'mini_batch']), 
              default='all', help='Gradient descent method to use')
@click.option('--sample-size', type=int, help='Sample size for faster testing')
@click.option('--learning-rate', type=float, default=0.01, help='Learning rate')
@click.option('--iterations', type=int, default=1000, help='Number of iterations')
@click.option('--batch-size', type=int, default=32, help='Batch size for mini-batch method')
@click.option('--use-custom', is_flag=True, default=True, help='Use custom optimized implementation')
@click.option('--regularization', type=click.Choice(['none', 'l1', 'l2', 'elastic']), 
              default='l2', help='Regularization type')
@click.option('--lambda-reg', type=float, default=0.01, help='Regularization strength')
def regression(method, sample_size, learning_rate, iterations, batch_size, use_custom, regularization, lambda_reg):
    """Run regression analysis with gradient descent variants."""
    from .models.regression import run_optimized_regression_pipeline, train_single_gradient_descent_method  
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    try:
        if method == 'all':
            # Import and run complete comparison pipeline
            
            
            console.print(Panel.fit(
                "[bold cyan]Hyper-Optimized Gradient Descent Comparison[/bold cyan]\n"
                f"Implementation: {'Custom + Numba' if use_custom else 'Sklearn Wrappers'}\n"
                f"Sample Size: {sample_size or 'Full Dataset'}",
                border_style="cyan"
            ))
            
            # Determine methods to test
            methods_to_test = ['batch', 'sgd', 'mini_batch']
            if use_custom:
                methods_to_test.append('adam')
            
            pipeline = run_optimized_regression_pipeline(
                sample_size=sample_size,
                use_custom=False,
                methods=methods_to_test
            )
            
            if pipeline:
                console.print(f"\n[bold green]🎉 Complete comparison finished![/bold green]")
                console.print(f"[green]📊 Check {REPORTS_DIR} for detailed reports and visualizations[/green]")
            else:
                console.print("[red]❌ Pipeline failed to complete[/red]")
        
        else:
            
            console.print(Panel.fit(
                f"[bold cyan]Training {method.upper()} Gradient Descent[/bold cyan]\n"
                f"Implementation: {'Custom Optimized' if use_custom else 'Sklearn'}\n"
                f"Learning Rate: {learning_rate}, Iterations: {iterations}",
                border_style="cyan"
            ))
            
            # Configure hyperparameters
            config = {
                'learning_rate': learning_rate,
                'n_iterations': iterations,
                'batch_size': batch_size if method == 'mini_batch' else 32,
                'regularization': regularization,
                'lambda_reg': lambda_reg,
                'use_custom': False
            }
            
            result = train_single_gradient_descent_method(method, **config)
            
            if result:
                console.print(f"\n[bold green]✅ {method.upper()} training completed successfully![/bold green]")
                
                # Display key results
                metrics = result.get('metrics', {})
                conv_info = result.get('convergence_info', {})
                
                console.print("\n[bold]Key Results:[/bold]")
                console.print(f"• Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
                console.print(f"• Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
                console.print(f"• Training Time: {conv_info.get('training_time', 0):.2f}s")
                console.print(f"• Converged: {'✓' if conv_info.get('converged', False) else '✗'}")
                
                console.print(f"\n[cyan]💡 For full comparison, use: --method all[/cyan]")
            else:
                console.print(f"[red]❌ {method.upper()} training failed[/red]")
    
    except ImportError as e:
        console.print(f"[red]❌ Failed to import regression modules: {e}[/red]")
        console.print("[yellow]Make sure all dependencies are installed and preprocessing is complete[/yellow]")
    
    except Exception as e:
        console.print(f"[red]❌ Regression analysis failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command()
@click.option('--user-id', type=int, required=True, help='User ID to get recommendations for')
@click.option('--n-recommendations', type=int, default=5, help='Number of recommendations')
@click.option('--method', type=click.Choice(['association', 'collaborative', 'hybrid']), 
              default='association', help='Recommendation method')
def recommend(user_id, n_recommendations, method):
    """Get movie recommendations for a specific user."""
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    DataCleaner = get_data_cleaner()
    # Load data
    cleaner = DataCleaner()
    ratings_df, movies_df, _ = cleaner.load_data()
    
    # Check if user exists
    if user_id not in ratings_df['userId'].unique():
        console.print(f"[red]User {user_id} not found in dataset[/red]")
        return
    
    if method == 'association':
        from .models.association import AssociationRuleMiner
        
        # Get user's watched movies
        user_movies = ratings_df[
            (ratings_df['userId'] == user_id) & 
            (ratings_df['rating'] >= 4.0)
        ]['movieId'].tolist()
        
        movie_map = dict(zip(movies_df['movieId'], movies_df['title']))
        watched_titles = [movie_map.get(mid, f"Movie_{mid}") for mid in user_movies]
        
        console.print(f"\n[cyan]User {user_id} has watched {len(watched_titles)} highly-rated movies[/cyan]")
        
        # Load or create association rules
        miner = AssociationRuleMiner()
        if not miner.load_results():
            console.print("[yellow]Creating association rules (this may take a while)...[/yellow]")
            miner.prepare_transactions(ratings_df, movies_df)
            miner.fit_fpgrowth()
            miner.save_results()
        
        # Get recommendations
        recommendations = miner.get_recommendations(watched_titles, n_recommendations)
        
        if recommendations:
            console.print(f"\n[bold]Top {n_recommendations} Recommendations:[/bold]")
            table = Table(box="rounded")
            table.add_column("Movie", style="cyan")
            table.add_column("Confidence", justify="right", style="green")
            
            for movie, confidence in recommendations:
                table.add_row(movie, f"{confidence:.3f}")
            
            console.print(table)
        else:
            console.print("[yellow]No recommendations found based on association rules[/yellow]")
    
    elif method == 'collaborative':
        from .models.collaborative.user_based import UserBasedCF
        from .models.collaborative.item_based import ItemBasedCF
        
        console.print("[cyan]Loading collaborative filtering models...[/cyan]")
        
        # Try to load pre-trained models
        user_cf_path = PROCESSED_DATA_DIR / "user_based_cf_model.pkl.gz"
        item_cf_path = PROCESSED_DATA_DIR / "item_based_cf_model.pkl.gz"
        
        if user_cf_path.exists():
            # Load user-based CF
            user_cf = UserBasedCF()
            user_cf.load_model(user_cf_path)
            
            # Get recommendations
            recommendations = user_cf.recommend_movies(user_id, n_recommendations)
            
            if recommendations:
                console.print(f"\n[bold]Top {n_recommendations} Recommendations (User-Based CF):[/bold]")
                table = Table(box="rounded")
                table.add_column("Movie", style="cyan")
                table.add_column("Predicted Rating", justify="right", style="green")
                
                movie_map = dict(zip(movies_df['movieId'], movies_df['title']))
                for movie_id, rating in recommendations:
                    movie_title = movie_map.get(movie_id, f"Movie {movie_id}")
                    table.add_row(movie_title, f"{rating:.2f}")
                
                console.print(table)
            else:
                console.print("[yellow]No recommendations found using collaborative filtering[/yellow]")
        else:
            console.print("[yellow]Collaborative filtering models not found. Training new model...[/yellow]")
            
            # Train a simple user-based CF model
            user_cf = UserBasedCF(k_neighbors=50)
            user_cf.fit(ratings_df)
            
            # Get recommendations
            recommendations = user_cf.recommend_movies(user_id, n_recommendations)
            
            if recommendations:
                console.print(f"\n[bold]Top {n_recommendations} Recommendations:[/bold]")
                table = Table(box="rounded")
                table.add_column("Movie", style="cyan")
                table.add_column("Predicted Rating", justify="right", style="green")
                
                movie_map = dict(zip(movies_df['movieId'], movies_df['title']))
                for movie_id, rating in recommendations:
                    movie_title = movie_map.get(movie_id, f"Movie {movie_id}")
                    table.add_row(movie_title, f"{rating:.2f}")
                
                console.print(table)
            
            # Save the model for future use
            user_cf.save_model()
    
    else:
        console.print(f"[yellow]{method} recommendation method not yet implemented[/yellow]")


@cli.command()
@click.option('--method', type=click.Choice(['kmeans', 'agglomerative', 'both']), 
              default='both', help='Clustering method for user segmentation')
@click.option('--n-clusters', type=int, default=8, help='Number of user clusters')
@click.option('--save-plots', is_flag=True, help='Save visualization plots')
def user_segmentation(method, n_clusters, save_plots):
    """Run user segmentation based on rating behaviors."""
    from .models.clustering import run_user_segmentation_pipeline
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]User Segmentation Analysis[/bold cyan]\n"
        f"Method: {method}, Clusters: {n_clusters}",
        border_style="cyan"
    ))
    
    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, _ = cleaner.load_data()
    
    # Determine methods to use
    if method == 'both':
        methods = ['kmeans', 'agglomerative']
    else:
        methods = [method]
    
    # Run segmentation
    results = run_user_segmentation_pipeline(
        ratings_df, movies_df,
        n_clusters=n_clusters,
        methods=methods
    )
    
    # Display comparative results if both methods were used
    if len(results) > 1:
        console.print("\n[bold]Method Comparison:[/bold]")
        comparison_table = Table(box="rounded")
        comparison_table.add_column("Method", style="cyan")
        comparison_table.add_column("Clusters", justify="right")
        comparison_table.add_column("Largest Cluster", justify="right")
        comparison_table.add_column("Smallest Cluster", justify="right")
        
        for method_name, segmentation in results.items():
            cluster_sizes = segmentation.user_features['cluster'].value_counts()
            comparison_table.add_row(
                method_name.title(),
                str(len(cluster_sizes)),
                f"{cluster_sizes.max():,} users",
                f"{cluster_sizes.min():,} users"
            )
        
        console.print(comparison_table)


@cli.command()
@click.option('--linkage', type=click.Choice(['ward', 'complete', 'average', 'single', 'all']), 
              default='ward', help='Linkage method for hierarchical clustering')
@click.option('--n-clusters', type=int, help='Number of movie clusters (auto if not specified)')
@click.option('--use-tags', is_flag=True, help='Include tag features in clustering')
@click.option('--plot-dendrogram', is_flag=True, help='Generate dendrogram visualization')
def movie_clustering(linkage, n_clusters, use_tags, plot_dendrogram):
    """Run hierarchical clustering on movies."""
    from .models.clustering import run_movie_clustering_pipeline
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]Hierarchical Movie Clustering[/bold cyan]\n"
        f"Linkage: {linkage}, Clusters: {n_clusters or 'Auto'}",
        border_style="cyan"
    ))
    
    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()
    
    if use_tags and tags_df is None:
        console.print("[yellow]Warning: No tags data available, proceeding without tags[/yellow]")
        use_tags = False
    
    # Determine linkage methods
    if linkage == 'all':
        linkage_methods = ['ward', 'complete', 'average', 'single']
    else:
        linkage_methods = [linkage]
    
    # Run clustering
    results = run_movie_clustering_pipeline(
        ratings_df, movies_df,
        tags_df=tags_df if use_tags else None,
        linkage_methods=linkage_methods,
        n_clusters=n_clusters
    )
    
    # Show sample movies from best performing method
    if len(results) > 1:
        console.print("\n[yellow]Use --linkage <method> to see detailed results for a specific method[/yellow]")


@cli.command()
@click.option('--compare-methods', is_flag=True, help='Compare different clustering methods')
@click.option('--cross-analysis', is_flag=True, help='Analyze user-movie cluster interactions')
def clustering(compare_methods, cross_analysis):
    """Run complete clustering analysis (users and movies)."""
    from .models.clustering import run_complete_clustering_analysis
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]Complete Clustering Analysis[/bold cyan]\n"
        "User Segmentation + Hierarchical Movie Clustering",
        border_style="cyan"
    ))
    
    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()
    
    # Check if we need to sample for performance
    if len(ratings_df) > 5_000_000:
        if Confirm.ask(f"Dataset has {len(ratings_df):,} ratings. Sample for faster processing?", default=True):
            sample_size = 2_000_000
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)
            console.print(f"[yellow]Sampled {sample_size:,} ratings for analysis[/yellow]")
    
    # Run complete analysis
    results = run_complete_clustering_analysis(ratings_df, movies_df, tags_df)
    
    # Generate summary report
    console.print("\n[bold]Clustering Analysis Summary:[/bold]")
    
    # User segmentation summary
    user_methods = list(results['user_segmentation'].keys())
    console.print(f"\n[cyan]User Segmentation:[/cyan]")
    console.print(f"  Methods tested: {', '.join(user_methods)}")
    console.print(f"  Clusters found: {results['user_segmentation'][user_methods[0]].n_clusters}")
    
    # Movie clustering summary  
    movie_methods = list(results['movie_clustering'].keys())
    console.print(f"\n[cyan]Movie Clustering:[/cyan]")
    console.print(f"  Linkage methods tested: {', '.join(movie_methods)}")
    
    if cross_analysis and 'patterns' in results['cross_analysis']:
        patterns = results['cross_analysis']['patterns']
        console.print(f"\n[cyan]Cross-Analysis:[/cyan]")
        console.print(f"  Interesting patterns found: {len(patterns)}")
        console.print(f"  See heatmap in: {REPORTS_DIR / 'user_movie_cluster_heatmap.png'}")


@cli.command()
@click.option('--user-cluster', type=int, help='User cluster ID')
@click.option('--movie-cluster', type=int, help='Movie cluster ID')
@click.option('--top-n', type=int, default=10, help='Number of items to show')
def cluster_details(user_cluster, movie_cluster, top_n):
    """Show details about specific clusters."""
    import gzip
    import pickle
    
    results_path = PROCESSED_DATA_DIR / 'clustering_results.pkl.gz'
    
    if not results_path.exists():
        console.print("[red]No clustering results found. Run 'clustering' command first.[/red]")
        return
    
    # Load results
    with gzip.open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    if user_cluster is not None:
        # Show user cluster details
        if 'user_segmentation' not in results or 'kmeans' not in results['user_segmentation']:
            console.print("[red]User segmentation results not found[/red]")
            return
        
        segmentation = results['user_segmentation']['kmeans']
        
        if user_cluster >= segmentation.n_clusters:
            console.print(f"[red]Invalid user cluster ID. Valid range: 0-{segmentation.n_clusters-1}[/red]")
            return
        
        # Get cluster info
        cluster_users = segmentation.user_features[segmentation.user_features['cluster'] == user_cluster]
        cluster_label = segmentation.cluster_labels[user_cluster][1]
        
        console.print(f"\n[bold]User Cluster {user_cluster}: {cluster_label}[/bold]")
        console.print(f"Size: {len(cluster_users):,} users")
        
        # Show cluster statistics
        stats_table = Table(box="rounded")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Mean", justify="right")
        stats_table.add_column("Std Dev", justify="right")
        
        for col in ['rating_mean', 'rating_count', 'movieId_nunique', 'rating_frequency']:
            if col in cluster_users.columns:
                mean_val = cluster_users[col].mean()
                std_val = cluster_users[col].std()
                stats_table.add_row(
                    col.replace('_', ' ').title(),
                    f"{mean_val:.2f}",
                    f"{std_val:.2f}"
                )
        
        console.print(stats_table)
    
    if movie_cluster is not None:
        # Show movie cluster details
        if 'movie_clustering' not in results or 'ward' not in results['movie_clustering']:
            console.print("[red]Movie clustering results not found[/red]")
            return
        
        clustering = results['movie_clustering']['ward']
        
        console.print(f"\n[bold]Movie Cluster {movie_cluster}[/bold]")
        
        # Get sample movies
        sample_movies = clustering.get_cluster_movies(movie_cluster, n=top_n)
        
        if not sample_movies.empty:
            movies_table = Table(box="rounded")
            movies_table.add_column("Title", style="cyan")
            movies_table.add_column("Year", justify="right")
            movies_table.add_column("Avg Rating", justify="right")
            movies_table.add_column("# Ratings", justify="right")
            
            for _, movie in sample_movies.iterrows():
                movies_table.add_row(
                    movie['title'][:50] + ('...' if len(movie['title']) > 50 else ''),
                    str(int(movie['year'])) if pd.notna(movie['year']) else 'N/A',
                    f"{movie['rating_mean']:.2f}",
                    f"{int(movie['rating_count']):,}"
                )
            
            console.print(movies_table)

@cli.command()
@click.option('--model', type=click.Choice(['rating', 'genre', 'user_type', 'all']), default='all', help='Classification model to run')
@click.option('--sample-size', type=int, help='Sample size for faster testing')
@click.option('--n-jobs', type=int, default=4, help='Number of parallel jobs')
def classification(model, sample_size, n_jobs):
    """Run classification models on the dataset."""
    from .models.classification import (
        RatingClassifier,
        GenrePredictor,
        UserTypeClassifier,
        run_classification_pipeline
    )
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return

    console.print(Panel.fit(
        "[bold cyan]Classification Models[/bold cyan]\n"
        f"Model: {model}, Sample Size: {sample_size or 'Full'}, Jobs: {n_jobs}",
        border_style="cyan"
    ))

    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()

    # Run all models pipeline
    if model == 'all':
        results = run_classification_pipeline(
            ratings_df=ratings_df,
            movies_df=movies_df,
            tags_df=tags_df,
            sample_size=sample_size,
            n_jobs=n_jobs
        )
        console.print("\n[bold]Classification Results:[/bold]")
        for name, res in results.items():
            console.print(f"[cyan]{name}[/cyan]: {res}")
        return

    # Run individual models
    if model == 'rating':
        clf = RatingClassifier(n_jobs=n_jobs)
        clf.fit(ratings_df, movies_df, sample_size=sample_size)
        metrics = clf.evaluate()
        console.print(f"\n[bold]Rating Classifier Results:[/bold] {metrics}")
    elif model == 'genre':
        clf = GenrePredictor(n_jobs=n_jobs)
        clf.fit(ratings_df, movies_df, sample_size=sample_size)
        metrics = clf.evaluate()
        console.print(f"\n[bold]Genre Predictor Results:[/bold] {metrics}")
    elif model == 'user_type':
        clf = UserTypeClassifier(n_jobs=n_jobs)
        clf.fit(ratings_df, movies_df, sample_size=sample_size)
        metrics = clf.evaluate()
        console.print(f"\n[bold]User Type Classifier Results:[/bold] {metrics}")


@cli.command()
@click.option('--method', type=click.Choice(['user', 'item', 'both']), 
              default='both', help='Collaborative filtering method')
@click.option('--k-neighbors', type=int, default=50, help='Number of neighbors to consider')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
def collaborative(method, k_neighbors, test_size):
    """Run collaborative filtering recommender systems."""
    from .models.collaborative.user_based import UserBasedCF, run_user_based_cf_pipeline
    from .models.collaborative.item_based import ItemBasedCF, run_collaborative_filtering_comparison
    
    # Check if preprocessed data exists
    ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    if not ml_datasets_path.exists():
        console.print("[red]❌ Preprocessed data not found. Please run 'python analyze.py preprocess' first.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold cyan]Collaborative Filtering[/bold cyan]\n"
        f"Method: {method}, K-Neighbors: {k_neighbors}",
        border_style="cyan"
    ))
    
    # Load data
    DataCleaner = get_data_cleaner()
    cleaner = DataCleaner()
    ratings_df, movies_df, _ = cleaner.load_data()
    
    if method == 'both':
        # Run comparison
        results = run_collaborative_filtering_comparison(
            ratings_df, 
            test_size=test_size
        )
    elif method == 'user':
        # Run user-based CF
        results = run_user_based_cf_pipeline(
            ratings_df,
            test_size=test_size,
            k_neighbors=[k_neighbors],
            similarity_metrics=['cosine']
        )
    else:  # item
        # Run item-based CF
        from sklearn.model_selection import train_test_split
        
        train_ratings, test_ratings = train_test_split(
            ratings_df, test_size=test_size, random_state=42
        )
        
        console.print("[cyan]Training Item-Based Collaborative Filtering...[/cyan]")
        item_cf = ItemBasedCF(k_neighbors=k_neighbors)
        item_cf.fit(train_ratings)
        
        console.print("[cyan]Evaluating model...[/cyan]")
        metrics = item_cf.evaluate(test_ratings)
        
        item_cf.save_model()
        results = {'item_based': {'model': item_cf, 'metrics': metrics}}
    
    console.print("\n[bold green]✓ Collaborative filtering complete![/bold green]")


# Helper functions

def create_ml_datasets_hyper_optimized(ratings_df, features, movies_df, n_jobs):
    """Create ML datasets using hyper-optimized processing."""
    
    console.print("[cyan]Creating ML datasets with hyper-optimization...[/cyan]")
    ml_datasets = {}
    
    with ProcessPoolExecutor(max_workers=min(n_jobs, 5)) as executor:
        # Submit all tasks
        futures = {
            'regression': executor.submit(create_regression_dataset_optimized, 
                                        ratings_df, features['user_features'], features['movie_features']),
            'classification': executor.submit(create_classification_dataset_optimized, 
                                            ratings_df, features['user_features'], features['movie_features']),
            'clustering_users': executor.submit(create_user_clustering_dataset_optimized, 
                                               features['user_features']),
            'clustering_movies': executor.submit(create_movie_clustering_dataset_optimized, 
                                                features['movie_features']),
            'association_rules': executor.submit(create_association_dataset_optimized, 
                                               ratings_df, movies_df)
        }
        
        # Collect results
        for task, future in futures.items():
            try:
                ml_datasets[task] = future.result()
                console.print(f"[green]✓ Created {task} dataset[/green]")
            except Exception as e:
                console.print(f"[red]✗ Error creating {task} dataset: {e}[/red]")
                ml_datasets[task] = None
    
    return ml_datasets

def create_regression_dataset_optimized(ratings_df, user_features, movie_features):
    """Create optimized regression dataset."""
    # Merge features using optimized joins
    feature_matrix = ratings_df.merge(
        user_features.add_suffix('_user'), 
        left_on='userId', 
        right_index=True,
        how='left'
    ).merge(
        movie_features.add_suffix('_movie'), 
        left_on='movieId', 
        right_index=True,
        how='left'
    )
    
    # Select numeric features only
    feature_cols = [col for col in feature_matrix.columns 
                   if col not in ['rating', 'userId', 'movieId', 'timestamp'] 
                   and feature_matrix[col].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    X = feature_matrix[feature_cols].fillna(0).astype(np.float32)
    y = feature_matrix['rating'].astype(np.float32)
    
    return {'X': X, 'y': y, 'feature_names': feature_cols}

def create_classification_dataset_optimized(ratings_df, user_features, movie_features):
    """Create optimized classification dataset."""
    # Similar to regression but with binary target
    feature_matrix = ratings_df.merge(
        user_features.add_suffix('_user'), 
        left_on='userId', 
        right_index=True,
        how='left'
    ).merge(
        movie_features.add_suffix('_movie'), 
        left_on='movieId', 
        right_index=True,
        how='left'
    )
    
    # Select numeric features only
    feature_cols = [col for col in feature_matrix.columns 
                   if col not in ['rating', 'userId', 'movieId', 'timestamp']
                   and feature_matrix[col].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    X = feature_matrix[feature_cols].fillna(0).astype(np.float32)
    y = (feature_matrix['rating'] >= 4).astype(np.int32)
    
    return {'X': X, 'y': y, 'feature_names': feature_cols}

def create_user_clustering_dataset_optimized(user_features):
    """Create optimized user clustering dataset."""
    numeric_features = user_features.select_dtypes(include=[np.number]).astype(np.float32)
    return {
        'X': numeric_features.fillna(0),
        'feature_names': numeric_features.columns.tolist(),
        'user_ids': user_features.index.tolist()
    }

def create_movie_clustering_dataset_optimized(movie_features):
    """Create optimized movie clustering dataset."""
    numeric_features = movie_features.select_dtypes(include=[np.number]).astype(np.float32)
    return {
        'X': numeric_features.fillna(0),
        'feature_names': numeric_features.columns.tolist(),
        'movie_ids': movie_features.index.tolist()
    }

def create_association_dataset_optimized(ratings_df, movies_df):
    """Create optimized association rules dataset."""
    # Get high-rated movies per user
    high_ratings = ratings_df[ratings_df['rating'] >= 4.0]
    transactions = high_ratings.groupby('userId')['movieId'].apply(list).tolist()
    
    return {
        'transactions': transactions,
        'movies_df': movies_df,
        'min_support': 0.01,
        'min_confidence': 0.5
    }

def save_results_hyper_optimized(features, ml_datasets, n_jobs):
    """Save results using hyper-optimized I/O operations."""
    
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    console.print("[cyan]Saving results with hyper-optimization...[/cyan]")
    
    # Save features
    features_path = PROCESSED_DATA_DIR / "hyper_features.pkl.gz"
    with gzip.open(features_path, 'wb') as f:
        pickle.dump(features, f, protocol=4)
    console.print(f"[green]✓ Saved features to {features_path}[/green]")
    
    # Save ML datasets
    ml_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
    with gzip.open(ml_path, 'wb') as f:
        pickle.dump(ml_datasets, f, protocol=4)
    console.print(f"[green]✓ Saved ML datasets to {ml_path}[/green]")
    
    # Save sparse matrices separately for efficiency
    if 'user_item_matrix' in features:
        from scipy.sparse import save_npz
        sparse_path = PROCESSED_DATA_DIR / "user_item_matrix.npz"
        save_npz(sparse_path, features['user_item_matrix'])
        console.print(f"[green]✓ Saved sparse matrix to {sparse_path}[/green]")

def display_hyper_performance_metrics(features, ml_datasets, total_time, n_jobs):
    """Display performance metrics for hyper-optimized processing."""
    
    metrics_table = Table(title="Hyper-Optimized Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Details", style="dim")
    
    # Time metrics
    metrics_table.add_row("Total Time", f"{total_time:.2f} seconds", "Wall clock time")
    
    # Feature metrics
    if 'user_features' in features:
        user_count = len(features['user_features'])
        user_feature_count = len(features['user_features'].columns)
        metrics_table.add_row("Users Processed", f"{user_count:,}", f"{user_feature_count} features each")
    
    if 'movie_features' in features:
        movie_count = len(features['movie_features'])
        movie_feature_count = len(features['movie_features'].columns)
        metrics_table.add_row("Movies Processed", f"{movie_count:,}", f"{movie_feature_count} features each")
    
    if 'user_item_matrix' in features:
        matrix = features['user_item_matrix']
        sparsity = 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        metrics_table.add_row("Sparse Matrix", f"{matrix.shape[0]}×{matrix.shape[1]}", f"{sparsity:.2%} sparse")
    
    # ML dataset metrics
    if ml_datasets:
        total_samples = sum(
            data['X'].shape[0] for data in ml_datasets.values() 
            if data and 'X' in data and hasattr(data['X'], 'shape')
        )
        ready_datasets = sum(1 for data in ml_datasets.values() if data is not None)
        metrics_table.add_row("ML Datasets Ready", f"{ready_datasets}/5", f"{total_samples:,} total samples")
    
    # Performance metrics
    metrics_table.add_row("Parallel Workers", str(n_jobs), "CPU cores utilized")
    
    console.print(metrics_table)
    
    # Processing speed
    if 'user_features' in features and user_count > 0:
        users_per_second = user_count / total_time
        console.print(f"\n[bold green]🚀 Processing speed: {users_per_second:,.0f} users/second[/bold green]")

def display_features_summary(features):
    """Display summary of hyper-optimized features."""
    
    summary_table = Table(title="Hyper-Optimized Features Summary", box=box.ROUNDED)
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Shape/Size", style="green")
    summary_table.add_column("Type", style="yellow")
    summary_table.add_column("Memory", style="dim")
    
    for name, data in features.items():
        if isinstance(data, pd.DataFrame):
            shape = f"{data.shape[0]:,}×{data.shape[1]}"
            data_type = "DataFrame"
            memory = f"{data.memory_usage(deep=True).sum() / 1024**2:.1f}MB"
        elif hasattr(data, 'shape'):
            shape = f"{data.shape[0]:,}×{data.shape[1]}"
            data_type = type(data).__name__
            memory = f"{data.data.nbytes / 1024**2:.1f}MB" if hasattr(data, 'data') else "N/A"
        else:
            shape = str(len(data)) if hasattr(data, '__len__') else "N/A"
            data_type = type(data).__name__
            memory = "N/A"
        
        summary_table.add_row(name.replace('_', ' ').title(), shape, data_type, memory)
    
    console.print(summary_table)

def display_benchmark_results(results: dict):
    """Display benchmark comparison results."""
    
    console.print("\n[bold cyan]Hyper-Optimized Benchmark Results[/bold cyan]")
    
    benchmark_table = Table(box=box.ROUNDED)
    benchmark_table.add_column("Mode", style="cyan")
    benchmark_table.add_column("Avg Time (s)", justify="right")
    benchmark_table.add_column("Min Time (s)", justify="right")
    benchmark_table.add_column("Max Memory (GB)", justify="right")
    benchmark_table.add_column("Success Rate", justify="right")
    benchmark_table.add_column("Speedup", justify="right", style="green")
    
    # Calculate baseline (balanced mode)
    baseline_time = None
    if 'balanced' in results:
        successful_runs = [r for r in results['balanced'] if r['success']]
        if successful_runs:
            baseline_time = sum(r['time'] for r in successful_runs) / len(successful_runs)
    
    for mode, mode_results in results.items():
        successful_runs = [r for r in mode_results if r['success']]
        
        if successful_runs:
            avg_time = sum(r['time'] for r in successful_runs) / len(successful_runs)
            min_time = min(r['time'] for r in successful_runs)
            max_memory = max(r['memory_peak'] for r in successful_runs)
            success_rate = len(successful_runs) / len(mode_results) * 100
            
            speedup = ""
            if baseline_time and avg_time > 0:
                speedup_factor = baseline_time / avg_time
                speedup = f"{speedup_factor:.2f}x"
            
            benchmark_table.add_row(
                mode.title(),
                f"{avg_time:.2f}",
                f"{min_time:.2f}",
                f"{max_memory:.2f}",
                f"{success_rate:.0f}%",
                speedup
            )
        else:
            benchmark_table.add_row(
                mode.title(),
                "Failed",
                "Failed", 
                "N/A",
                "0%",
                "N/A"
            )
    
    console.print(benchmark_table)
    
    # Recommendations
    console.print("\n[bold cyan]Hyper-Optimization Recommendations[/bold cyan]")
    
    if results:
        best_speed_mode = min(
            results.keys(),
            key=lambda mode: min(
                (r['time'] for r in results[mode] if r['success']),
                default=float('inf')
            )
        )
        
        console.print(f"🚀 [green]Fastest mode: {best_speed_mode}[/green]")
        console.print(f"💾 [blue]Most memory efficient: memory mode[/blue]")
        console.print(f"⚖️ [yellow]Best balance: balanced mode[/yellow]")
        console.print(f"🔥 [red]All modes use hyper-optimization with numba and parallel processing[/red]")

class OptimizedPerformanceConfig:
    """Optimized configuration for 20M+ record datasets."""
    
    def __init__(self, available_memory_gb: float = None):
        import psutil
        
        # Auto-detect available memory if not provided
        if available_memory_gb is None:
            available_memory_gb = psutil.virtual_memory().available / 1024**3
        
        # Leave 20% memory buffer for system
        self.memory_limit_gb = available_memory_gb * 0.8
        
        # Optimized batch sizes based on available memory
        if self.memory_limit_gb < 8:
            # Low memory mode
            self.user_batch_size = 5_000
            self.movie_batch_size = 2_500
            self.chunk_size = 100_000
            self.rating_sample_size = 500_000
        elif self.memory_limit_gb < 16:
            # Medium memory mode
            self.user_batch_size = 10_000
            self.movie_batch_size = 5_000
            self.chunk_size = 500_000
            self.rating_sample_size = 1_000_000
        else:
            # High memory mode
            self.user_batch_size = 20_000
            self.movie_batch_size = 10_000
            self.chunk_size = 1_000_000
            self.rating_sample_size = 2_000_000
        
        # Parallel processing settings
        self.n_jobs = min(mp.cpu_count() - 1, 8)  # Leave one core free
        self.use_sample_for_clustering = True
        self.cluster_sample_size = min(50_000, self.chunk_size // 10)
        
        # Feature engineering settings
        self.skip_expensive_features = self.memory_limit_gb < 8
        self.use_sparse_matrices = True
        self.compress_intermediate_results = True


if __name__ == '__main__':
    cli()