import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from .config import PROCESSED_DATA_DIR
from .preprocessing.cleaner import DataCleaner
from .preprocessing.pipeline import PreprocessingPipeline
try:
    # Try to import the fixed GPU transformer first
    from .preprocessing.transformer_ultrafast import HyperOptimizedDataTransformer
    GPU_TRANSFORMER_AVAILABLE = True
except ImportError:
    # Fall back to regular transformer
    from .preprocessing.transformer import DataTransformer as HyperOptimizedDataTransformer
    GPU_TRANSFORMER_AVAILABLE = False
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
import gc


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
    
    
    def get_transformer(self) -> HyperOptimizedDataTransformer:
        """Get configured transformer instance."""
        return HyperOptimizedDataTransformer(
            n_jobs=self.n_jobs,
            use_gpu=GPU_TRANSFORMER_AVAILABLE
        )

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
        features = transformer.create_all_features_pipeline_gpu(
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
                
                # Create ML datasets (use fixed version if GPU available)
                if GPU_TRANSFORMER_AVAILABLE:
                    ml_datasets = create_ml_datasets_fixed(
                        ratings_sample, features, movies_df, config.n_jobs
                    )
                else:
                    ml_datasets = create_ml_datasets_hyper_optimized(
                        ratings_sample, features, movies_df, config.n_jobs
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

            if 'class_balance' in dataset and task == 'classification':
                info_table.add_row("Class Balance", f"Positive: {dataset['class_balance']['positive']:.2%}, Negative: {dataset['class_balance']['negative']:.2%}")

        elif task == 'association_rules':
            transactions = dataset.get('transactions', [])
            info_table.add_row("Type", "Association Rules")
            info_table.add_row("Transactions", f"{len(transactions):,}")
            info_table.add_row("Min Support", str(dataset.get('min_support')))
            info_table.add_row("Min Confidence", str(dataset.get('min_confidence')))

        if 'n_samples' in dataset:
            info_table.add_row("Confirmed Samples", f"{dataset['n_samples']:,}")
        if 'n_features' in dataset:
            info_table.add_row("Confirmed Features", f"{dataset['n_features']}")

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
    console.print("[yellow]Running basic data cleaning...[/yellow]")
    cleaner = DataCleaner()
    cleaner.load_data()
    cleaner.clean_ratings(save=False)
    cleaner.clean_movies(save=False)
    cleaner.clean_tags(save=False)
    console.print("[green]✓ Basic data cleaning demonstration completed (data not saved).[/green]")

@cli.command()
@click.option('--memory-limit', type=float, help='Memory limit in GB (auto-detected if not specified)')
@click.option('--ultra-fast', is_flag=True, help='Use ultra-fast mode (may reduce feature quality)')
def preprocess_optimized(memory_limit, ultra_fast):
    """Run optimized preprocessing for very large datasets."""
    
    # Create optimized configuration
    config = OptimizedPerformanceConfig(available_memory_gb=memory_limit)
    
    if ultra_fast:
        config.user_batch_size = min(config.user_batch_size, 5000)
        config.skip_expensive_features = True
        config.use_sample_for_clustering = True
        config.rating_sample_size = min(config.rating_sample_size, 500_000)
    
    console.print(Panel.fit(
        f"[bold blue]Hyper-Optimized Preprocessing Configuration[/bold blue]\n"
        f"Memory Limit: {config.memory_limit_gb:.1f} GB\n"
        f"User Batch Size: {config.user_batch_size:,}\n"
        f"Rating Sample Size: {config.rating_sample_size:,}\n"
        f"Parallel Workers: {config.n_jobs}\n"
        f"Mode: {'Ultra-Fast' if ultra_fast else 'Balanced'}",
        border_style="blue"
    ))
    
    # Use the hyper-optimized transformer
    transformer = HyperOptimizedDataTransformer(n_jobs=config.n_jobs)
    
    try:
        # Initialize data cleaner
        cleaner = DataCleaner()
        ratings_df, movies_df, tags_df = cleaner.load_data()
        
        # Clean data
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
        
        console.print("[bold green]✅ Hyper-optimized preprocessing completed![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

@cli.command()
@click.option('--use-gpu', is_flag=True, help='Use GPU acceleration')
@click.option('--batch-size', type=int, default=100000, help='Batch size for processing')
@click.option('--sample-size', type=int, default=1000000, help='Sample size for ML datasets')
def preprocess_gpu_fixed(use_gpu, batch_size, sample_size):
    """Run preprocessing optimized for 4GB VRAM GPUs."""
    
    console.print("\n[bold cyan]4GB VRAM Optimized Preprocessing[/bold cyan]")
    console.print(f"GPU: {'Enabled' if use_gpu and GPU_TRANSFORMER_AVAILABLE else 'Disabled'}")
    console.print(f"Batch size: {batch_size:,}")
    console.print(f"ML sample size: {sample_size:,}")
    
    start_time = time.time()
    
    try:
        # Initialize fixed transformer
        transformer = HyperOptimizedDataTransformer(
            use_gpu=use_gpu and GPU_TRANSFORMER_AVAILABLE,
            n_jobs=mp.cpu_count()
        )
        
        # Load data
        cleaner = DataCleaner()
        ratings_df, movies_df, tags_df = cleaner.load_data()
        
        # Clean data
        console.print("\n[cyan]Cleaning data...[/cyan]")
        cleaned_ratings = cleaner.clean_ratings(save=False, optimize_memory=True)
        cleaned_movies = cleaner.clean_movies(save=False)
        cleaned_tags = cleaner.clean_tags(save=False) if tags_df is not None else None
        
        # Run feature engineering
        console.print("\n[cyan]Creating features...[/cyan]")
        features = transformer.create_all_features_pipeline(
            cleaned_ratings, 
            cleaned_movies, 
            cleaned_tags,
            batch_size=batch_size
        )
        
        # Create ML datasets
        console.print("\n[cyan]Creating ML datasets...[/cyan]")
        ml_datasets = create_ml_datasets_fixed(
            cleaned_ratings, features, cleaned_movies, transformer.n_jobs
        )
        
        # Save results
        console.print("\n[cyan]Saving results...[/cyan]")
        save_results_hyper_optimized(features, ml_datasets, transformer.n_jobs)
        
        elapsed = time.time() - start_time
        console.print(f"\n[bold green]✅ Complete in {elapsed:.1f} seconds![/bold green]")
        
        # Show what was created
        for task, dataset in ml_datasets.items():
            if dataset and 'X' in dataset and hasattr(dataset['X'], 'shape'):
                console.print(f"{task.title()} dataset: {dataset['X'].shape}")
                if task == 'classification' and 'class_balance' in dataset:
                    console.print(f"Class balance: {dataset['class_balance']}")
            elif dataset and task == 'association_rules':
                console.print(f"Association rules: {dataset['n_transactions']:,} transactions")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        
    finally:
        # Cleanup
        gc.collect()
        if use_gpu and GPU_TRANSFORMER_AVAILABLE:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass


@cli.command()
def test_ml_datasets():
    """Test if ML datasets were created properly."""
    
    import gzip
    import pickle
    from pathlib import Path
    
    ml_path = Path("data/processed/ml_ready_datasets.pkl.gz")
    
    if not ml_path.exists():
        console.print("[red]ML datasets not found![/red]")
        return
    
    console.print("[cyan]Loading ML datasets...[/cyan]")
    
    try:
        with gzip.open(ml_path, 'rb') as f:
            ml_datasets = pickle.load(f)
        
        console.print("\n[bold]ML Datasets Status:[/bold]")
        
        for task, data in ml_datasets.items():
            if data is None:
                console.print(f"[yellow]{task}: Not created (None)[/yellow]")
            elif isinstance(data, dict) and 'X' in data:
                X = data['X']
                y = data['y']
                console.print(f"[green]{task}: {X.shape} samples, {y.shape} targets[/green]")
                
                if task == 'classification' and 'class_balance' in data:
                    balance = data['class_balance']
                    console.print(f"  Class balance: {balance}")
            else:
                console.print(f"[red]{task}: Invalid format[/red]")
                
    except Exception as e:
        console.print(f"[red]Error loading ML datasets: {e}[/red]")

# Update the preprocess command to use the fixed pipeline
@cli.command()
@click.option('--gpu-fixed', is_flag=True, help='Use 4GB VRAM optimized pipeline')
@click.option('--batch-size', type=int, default=200000)
@cli.command()
@click.option('--gpu-fixed', is_flag=True, help='Use 4GB VRAM optimized pipeline')
@click.option('--batch-size', type=int, default=200000)
def preprocess_4gb(gpu_fixed, batch_size):
    """Run preprocessing with 4GB VRAM optimization."""
    
    if gpu_fixed:
        # Use the pipeline-based approach
        try:
            from .preprocessing.pipeline import PreprocessingPipeline
        except ImportError:
            console.print("[red]❌ 4GB VRAM pipeline not available. Falling back to transformer-based approach.[/red]")
            gpu_fixed = False
    
    if gpu_fixed:
        pipeline = PreprocessingPipeline()
        pipeline.run_full_pipeline_with_monitoring(
            create_sparse_matrices=True,
            apply_pca=False,  # Skip PCA for memory
            save_results=True,
            use_cache=True,
            memory_limit_gb=12.0,  # System RAM limit
            validate_steps=False,  # Skip validation for speed
            batch_size=batch_size
        )
    else:
        # Use transformer-based approach with fixed ML dataset creation
        console.print("\n[bold cyan]4GB VRAM Optimized Preprocessing (Transformer-Based)[/bold cyan]")
        console.print(f"Batch size: {batch_size:,}")
        
        start_time = time.time()
        
        try:
            # Initialize components
            cleaner = DataCleaner()
            transformer = HyperOptimizedDataTransformer(use_gpu=GPU_TRANSFORMER_AVAILABLE, n_jobs=mp.cpu_count())
            
            # Load and clean data
            ratings_df, movies_df, tags_df = cleaner.load_data()
            cleaned_ratings = cleaner.clean_ratings(save=False, optimize_memory=True)
            cleaned_movies = cleaner.clean_movies(save=False)
            cleaned_tags = cleaner.clean_tags(save=False) if tags_df is not None else None
            
            # Run feature engineering
            console.print("[cyan]Running feature engineering...[/cyan]")
            features = transformer.create_all_features_pipeline(
                cleaned_ratings, cleaned_movies, cleaned_tags,
                batch_size=batch_size
            )
            
            # Create ML datasets
            console.print("[cyan]Creating ML datasets...[/cyan]")
            ml_datasets = create_ml_datasets_fixed(
                cleaned_ratings, features, cleaned_movies, transformer.n_jobs
            )
            
            # Save results
            console.print("[cyan]Saving results...[/cyan]")
            save_results_hyper_optimized(features, ml_datasets, transformer.n_jobs)
            
            end_time = time.time()
            total_time = end_time - start_time
            console.print(f"\n[bold green]✅ Completed in {total_time:.2f} seconds![/bold green]")
            
            # Display performance metrics
            display_hyper_performance_metrics(features, ml_datasets, total_time, transformer.n_jobs)
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Pipeline error: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
        
        finally:
            gc.collect()
            if GPU_TRANSFORMER_AVAILABLE:
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass

# Support functions for hyper-optimized processing
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


def create_ml_datasets_fixed(ratings_df, features, movies_df, n_jobs):
    """Fixed ML dataset creation for reliable regression/classification datasets."""
    from rich.console import Console
    console = Console()
    
    console.print("[cyan]Creating ML datasets (reliability-focused)...[/cyan]")
    ml_datasets = {}
    
    try:
        # Get and validate features
        user_features = features.get('user_features')
        movie_features = features.get('movie_features')
        
        if user_features is None or movie_features is None:
            raise ValueError("Missing user or movie features")
        
        # Validate features for ML use
        if hasattr(user_features, 'validate_features_for_ml'):
            # If transformer method available, use it
            user_clean, movie_clean = user_features.validate_features_for_ml(user_features, movie_features)
        else:
            # Manual validation
            console.print("[cyan]Manually validating features...[/cyan]")
            
            # Clean user features
            user_clean = user_features.select_dtypes(include=[np.number]).fillna(0)
            user_clean = user_clean.replace([np.inf, -np.inf], 0)
            
            # Clean movie features  
            movie_clean = movie_features.select_dtypes(include=[np.number]).fillna(0)
            movie_clean = movie_clean.replace([np.inf, -np.inf], 0)
        
        # Sample for memory efficiency
        sample_size = min(1_000_000, len(ratings_df))
        if len(ratings_df) > sample_size:
            console.print(f"[yellow]Sampling {sample_size:,} ratings for ML datasets[/yellow]")
            ratings_sample = ratings_df.sample(n=sample_size, random_state=42)
        else:
            ratings_sample = ratings_df.copy()
        
        # Start with core rating data
        console.print("[cyan]Building feature matrix...[/cyan]")
        feature_matrix = ratings_sample[['userId', 'movieId', 'rating']].copy()
        
        # Limit features to prevent memory issues
        max_features = 50
        
        # Add user features with careful merging
        user_feature_cols = user_clean.columns.tolist()[:max_features]
        console.print(f"[cyan]Adding {len(user_feature_cols)} user features...[/cyan]")
        
        # Create user feature mapping
        user_feature_dict = {}
        for col in user_feature_cols:
            user_feature_dict[f'user_{col}'] = user_clean[col].to_dict()
        
        # Map user features
        for feature_name, feature_dict in user_feature_dict.items():
            feature_matrix[feature_name] = feature_matrix['userId'].map(feature_dict).fillna(0)
        
        # Add movie features with careful merging
        movie_feature_cols = movie_clean.columns.tolist()[:max_features] 
        console.print(f"[cyan]Adding {len(movie_feature_cols)} movie features...[/cyan]")
        
        # Create movie feature mapping
        movie_feature_dict = {}
        for col in movie_feature_cols:
            movie_feature_dict[f'movie_{col}'] = movie_clean[col].to_dict()
        
        # Map movie features
        for feature_name, feature_dict in movie_feature_dict.items():
            feature_matrix[feature_name] = feature_matrix['movieId'].map(feature_dict).fillna(0)
        
        # Get final feature columns
        feature_cols = [col for col in feature_matrix.columns 
                       if col not in ['userId', 'movieId', 'rating']]
        
        console.print(f"[green]Created feature matrix with {len(feature_cols)} features[/green]")
        
        # Final validation of feature matrix
        feature_data = feature_matrix[feature_cols]
        
        # Ensure all data is numeric
        for col in feature_data.columns:
            if feature_data[col].dtype == 'object':
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
        
        # Fill any remaining issues
        feature_data = feature_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Convert to appropriate numpy arrays
        X = feature_data.values.astype(np.float32)
        y_reg = feature_matrix['rating'].values.astype(np.float32)
        y_clf = (feature_matrix['rating'] >= 4.0).astype(np.int32)
        
        # Validate shapes
        if X.shape[0] != len(y_reg):
            raise ValueError(f"Shape mismatch: X has {X.shape[0]} rows, y has {len(y_reg)} rows")
        
        # Create regression dataset
        ml_datasets['regression'] = {
            'X': X,
            'y': y_reg,
            'feature_names': feature_cols,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
        console.print(f"[green]✓ Regression dataset: {X.shape}[/green]")
        
        # Create classification dataset
        ml_datasets['classification'] = {
            'X': X,  # Reuse same features
            'y': y_clf,
            'feature_names': feature_cols,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'class_balance': {
                'positive': float(np.sum(y_clf == 1) / len(y_clf)),
                'negative': float(np.sum(y_clf == 0) / len(y_clf))
            }
        }
        console.print(f"[green]✓ Classification dataset: {X.shape}[/green]")
        
        # Create simplified clustering datasets
        console.print("[cyan]Creating clustering datasets...[/cyan]")
        
        # User clustering dataset (simplified)
        user_clustering_data = user_clean.iloc[:, :20].fillna(0).values.astype(np.float32)  # Limit to 20 features
        ml_datasets['clustering_users'] = {
            'X': user_clustering_data,
            'feature_names': user_clean.columns[:20].tolist(),
            'user_ids': user_clean.index.tolist(),
            'n_samples': len(user_clustering_data),
            'n_features': user_clustering_data.shape[1]
        }
        
        # Movie clustering dataset (simplified)
        movie_clustering_data = movie_clean.iloc[:, :20].fillna(0).values.astype(np.float32)  # Limit to 20 features
        ml_datasets['clustering_movies'] = {
            'X': movie_clustering_data,
            'feature_names': movie_clean.columns[:20].tolist(),
            'movie_ids': movie_clean.index.tolist(),
            'n_samples': len(movie_clustering_data),
            'n_features': movie_clustering_data.shape[1]
        }
        
        # Create association rules dataset
        console.print("[cyan]Creating association rules dataset...[/cyan]")
        high_ratings = ratings_sample[ratings_sample['rating'] >= 4.0]
        transactions = high_ratings.groupby('userId')['movieId'].apply(list).tolist()
        ml_datasets['association_rules'] = {
            'transactions': transactions,
            'movies_df': movies_df,
            'min_support': 0.01,
            'min_confidence': 0.5,
            'n_transactions': len(transactions)
        }
        
        console.print("[green]✓ Successfully created all ML datasets[/green]")
        
        # Clean up intermediate variables
        del feature_matrix, feature_data
        gc.collect()
        
    except Exception as e:
        console.print(f"[red]✗ Error creating ML datasets: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        
        # Return None datasets on error
        ml_datasets = {
            'regression': None,
            'classification': None, 
            'clustering_users': None,
            'clustering_movies': None,
            'association_rules': None
        }
    
    return ml_datasets


def create_regression_dataset_optimized(ratings_df, user_features, movie_features):
    """Create optimized regression dataset with proper validation."""
    try:
        # Validate inputs
        if user_features.empty or movie_features.empty:
            raise ValueError("Empty feature DataFrames")
        
        # Clean features first
        user_clean = user_features.select_dtypes(include=[np.number]).fillna(0)
        movie_clean = movie_features.select_dtypes(include=[np.number]).fillna(0)
        
        # Sample if too large
        if len(ratings_df) > 500000:
            ratings_sample = ratings_df.sample(n=500000, random_state=42)
        else:
            ratings_sample = ratings_df
        
        # Start with ratings
        feature_matrix = ratings_sample[['userId', 'movieId', 'rating']].copy()
        
        # Add user features via mapping (memory efficient)
        user_cols = user_clean.columns[:25]  # Limit features
        for col in user_cols:
            feature_matrix[f'user_{col}'] = feature_matrix['userId'].map(
                user_clean[col].to_dict()
            ).fillna(0)
        
        # Add movie features via mapping  
        movie_cols = movie_clean.columns[:25]  # Limit features
        for col in movie_cols:
            feature_matrix[f'movie_{col}'] = feature_matrix['movieId'].map(
                movie_clean[col].to_dict()
            ).fillna(0)
        
        # Get feature columns
        feature_cols = [col for col in feature_matrix.columns 
                       if col not in ['rating', 'userId', 'movieId']]
        
        X = feature_matrix[feature_cols].values.astype(np.float32)
        y = feature_matrix['rating'].values.astype(np.float32)
        
        return {'X': X, 'y': y, 'feature_names': feature_cols}
        
    except Exception as e:
        print(f"Error in regression dataset: {e}")
        return None


def create_classification_dataset_optimized(ratings_df, user_features, movie_features):
    """Create optimized classification dataset with proper validation."""
    try:
        # Reuse regression logic but with binary target
        regression_data = create_regression_dataset_optimized(ratings_df, user_features, movie_features)
        
        if regression_data is None:
            return None
        
        # Create binary target (4+ stars = positive)
        y_binary = (regression_data['y'] >= 4.0).astype(np.int32)
        
        return {
            'X': regression_data['X'],
            'y': y_binary,
            'feature_names': regression_data['feature_names']
        }
        
    except Exception as e:
        print(f"Error in classification dataset: {e}")
        return None
    
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