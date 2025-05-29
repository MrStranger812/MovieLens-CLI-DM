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
from .preprocessing.transformer import DataTransformer
import pickle
import gzip
import multiprocessing as mp
import psutil
import time

console = Console()

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
    
    def get_transformer(self) -> DataTransformer:
        """Get configured transformer instance."""
        return DataTransformer(n_jobs=self.n_jobs)

@click.group()
def cli():
    """MovieLens Multi-Analytics CLI - Enhanced Performance Edition"""
    console.print(Panel.fit(
        "[bold blue]MovieLens Multi-Analytics Project[/bold blue]\n"
        "Comprehensive data science analysis tool with performance optimizations\n"
        f"System: {mp.cpu_count()} cores, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM",
        border_style="blue"
    ))

@cli.command()
@click.option('--technique', type=click.Choice(['regression', 'classification', 'clustering', 'recommender', 'association', 'all']),
              default='all', help='Analysis technique to run')
def analyze(technique):
    """Run data analysis with specified technique.""" 
    console.print(f"[yellow]Placeholder: Starting {technique} analysis...[/yellow]")
    console.print("[yellow]Note: Analysis functions need to be implemented.[/yellow]")

@cli.command()
def explore():
    """Explore the basic dataset info."""
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
def preprocess(skip_pca, skip_sparse, no_save, no_cache, clear_cache, memory_limit, 
               no_validation, batch_size, n_jobs):
    """Run optimized preprocessing pipeline with performance tuning."""
    
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
    system_table.add_row("Batch Size", f"{batch_size:,}", "Parallel processing")
    
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

    if not Confirm.ask("Proceed with preprocessing?", default=True):
        console.print("[yellow]Preprocessing cancelled.[/yellow]")
        return
    
    # Start timing
    start_time = time.time()
    
    # Initialize pipeline with parallel processing
    pipeline = PreprocessingPipeline(n_jobs=actual_n_jobs)
    
    try:
        console.print("\n[bold green]Starting Parallel Preprocessing Pipeline[/bold green]")
        
        results = pipeline.run_full_pipeline_with_monitoring(
            create_sparse_matrices=not skip_sparse,
            apply_pca=not skip_pca,
            save_results=not no_save,
            use_cache=not no_cache,
            memory_limit_gb=memory_limit,
            validate_steps=not no_validation,
            batch_size=batch_size
        )
        
        if results:
            end_time = time.time()
            total_time = end_time - start_time
            
            console.print(f"\n[bold green]✅ Pipeline completed in {total_time:.2f} seconds![/bold green]")
            
            # Display performance metrics
            display_performance_metrics(results, total_time, actual_n_jobs, batch_size)
            
        else:
            console.print("\n[bold red]❌ Pipeline failed![/bold red]")
            
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
def preprocess_fast(performance_mode, n_jobs, batch_size, memory_limit, 
                   skip_validation, profile):
    """Run optimized preprocessing pipeline with performance tuning."""
    
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
    system_table.add_row("Mode", performance_mode.title(), "Optimized processing")
    
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
    console.print("\n[bold cyan]Performance Configuration[/bold cyan]")
    config_table = Table(box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    
    config_table.add_row("User Batch Size", f"{config.user_batch_size:,}")
    config_table.add_row("Movie Batch Size", f"{config.movie_batch_size:,}")
    config_table.add_row("Chunk Size", f"{config.chunk_size:,}")
    config_table.add_row("TF-IDF Features", str(config.max_tfidf_features))
    config_table.add_row("Skip Validation", str(config.skip_validation))
    config_table.add_row("Workers", str(config.n_jobs))
    
    console.print(config_table)
    
    if not Confirm.ask("Proceed with these settings?", default=True):
        console.print("[yellow]Preprocessing cancelled.[/yellow]")
        return
    
    # Start timing
    start_time = time.time()
    
    # Initialize components
    pipeline = PreprocessingPipeline(n_jobs=config.n_jobs)
    
    # Enable profiling if requested
    if profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        console.print("\n[bold green]Starting Optimized Preprocessing Pipeline[/bold green]")
        
        results = pipeline.run_full_pipeline_with_monitoring(
            create_sparse_matrices=True,
            apply_pca=not skip_validation,
            save_results=True,
            use_cache=True,
            memory_limit_gb=config.memory_limit_gb,
            validate_steps=not (skip_validation or config.skip_validation),
            large_dataset_threshold=config.chunk_size,
            sparse_tfidf_threshold=config.chunk_size // 2,
            batch_size=config.user_batch_size
        )
        
        if results:
            end_time = time.time()
            total_time = end_time - start_time
            
            console.print(f"\n[bold green]✅ Pipeline completed in {total_time:.2f} seconds![/bold green]")
            
            # Display performance metrics
            display_performance_metrics(results, total_time, config.n_jobs, config.user_batch_size)
            
        else:
            console.print("\n[bold red]❌ Pipeline failed![/bold red]")
            
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
    
    console.print("[bold cyan]Starting Preprocessing Benchmarks[/bold cyan]")
    
    modes_list = [mode.strip() for mode in modes.split(',')]
    results = {}
    
    for mode in modes_list:
        console.print(f"\n[yellow]Benchmarking {mode} mode...[/yellow]")
        
        mode_results = []
        
        for i in range(iterations):
            console.print(f"[dim]Iteration {i+1}/{iterations}[/dim]")
            
            # Create configuration
            config = PerformanceConfig(mode=mode)
            
            # Time the preprocessing
            start_time = time.time()
            
            try:
                pipeline = PreprocessingPipeline(n_jobs=config.n_jobs)
                
                # Run a subset of preprocessing for benchmarking
                result = pipeline.run_benchmark_subset(config, None)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if result:
                    mode_results.append({
                        'time': elapsed_time,
                        'memory_peak': psutil.Process().memory_info().rss / 1024**3,
                        'success': True
                    })
                else:
                    mode_results.append({
                        'time': float('inf'),
                        'memory_peak': 0,
                        'success': False
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
    
    pipeline = PreprocessingPipeline()
    cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
    ml_datasets_gz_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"

    if cache_file.exists():
        console.print(f"[yellow]Loading cache: {cache_file}...[/yellow]")
        try:
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)

            if 'pipeline_metrics' in cached_results:
                console.print("[green]✓ Loaded cache with metrics. Displaying summary...[/green]")
                pipeline._display_enhanced_summary(cached_results, cached_results['pipeline_metrics'])
            else:
                console.print("[yellow]Cache exists but no metrics found. Showing basic info...[/yellow]")
                console.print(f"Cache keys: {list(cached_results.keys())}")

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
            "[yellow]💡 Tip: Run 'python analyze.py preprocess' to create processed datasets[/yellow]",
            border_style="yellow"
        ))

@cli.command('get-dataset')
@click.argument('task', type=click.Choice(['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']))
def get_dataset(task):
    """Get information about a specific preprocessed dataset."""
    
    pipeline = PreprocessingPipeline()
    dataset = pipeline.get_dataset_for_task(task)

    if not dataset:
        console.print(f"[red]Dataset for {task} not found. Run 'preprocess' first.[/red]")
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

@cli.command()
def validate():
    """Validate all preprocessed ML datasets."""
    
    pipeline = PreprocessingPipeline()
    all_tasks = ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']
    validation_table = Table(title="Data Validation Results", box=box.ROUNDED)
    validation_table.add_column("ML Task", style="cyan")
    validation_table.add_column("Status", style="bold")
    validation_table.add_column("Details", style="dim")

    found_any = False
    for task_name in all_tasks:
        dataset = pipeline.get_dataset_for_task(task_name)

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

def display_performance_metrics(results: dict, total_time: float, n_jobs: int, batch_size: int):
    """Display detailed performance metrics."""
    
    metrics_table = Table(title="Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Details", style="dim")
    
    # Time metrics
    metrics_table.add_row("Total Time", f"{total_time:.2f} seconds", "Wall clock time")
    
    if 'pipeline_metrics' in results:
        metrics = results['pipeline_metrics']
        
        if 'steps_completed' in metrics:
            steps_per_second = len(metrics['steps_completed']) / total_time
            metrics_table.add_row("Processing Rate", f"{steps_per_second:.2f} steps/sec", "Overall throughput")
        
        if 'memory_usage' in metrics and metrics['memory_usage']:
            peak_memory = max(metrics['memory_usage'])
            avg_memory = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
            metrics_table.add_row("Peak Memory", f"{peak_memory:.2f} GB", "Maximum memory usage")
            metrics_table.add_row("Avg Memory", f"{avg_memory:.2f} GB", "Average memory usage")
    
    # Dataset metrics
    if 'ml_ready_datasets' in results:
        datasets = results['ml_ready_datasets']
        total_samples = sum(
            data['X'].shape[0] for data in datasets.values() 
            if data and 'X' in data and hasattr(data['X'], 'shape')
        )
        metrics_table.add_row("Total Samples", f"{total_samples:,}", "Across all ML datasets")
    
    # Configuration metrics
    metrics_table.add_row("Parallel Jobs", str(n_jobs), "CPU cores utilized")
    metrics_table.add_row("Batch Size", f"{batch_size:,}", "Processing batch size")
    
    console.print(metrics_table)

def display_benchmark_results(results: dict):
    """Display benchmark comparison results."""
    
    console.print("\n[bold cyan]Benchmark Results Summary[/bold cyan]")
    
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
    console.print("\n[bold cyan]Recommendations[/bold cyan]")
    
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

if __name__ == '__main__':
    cli()