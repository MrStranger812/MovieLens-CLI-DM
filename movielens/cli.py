import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from .config import PROCESSED_DATA_DIR
from .preprocessing.cleaner import DataCleaner
from .preprocessing.pipeline import PreprocessingPipeline
import pickle
import gzip

console = Console()

@click.group()
def cli():
    """MovieLens Multi-Analytics CLI"""
    console.print(Panel.fit(
        "[bold blue]MovieLens Multi-Analytics Project[/bold blue]\n"
        "Comprehensive data science analysis tool",
        border_style="blue"
    ))

@cli.command()
@click.option('--technique', type=click.Choice(['regression', 'classification', 'clustering', 'recommender', 'association', 'all']),
              default='all', help='Analysis technique to run (Placeholder)')
def analyze(technique):
    """Run data analysis with specified technique."""
    console.print(f"[yellow]Placeholder: Starting {technique} analysis...[/yellow]")
    # --- Add analysis logic here ---
    console.print("[yellow]Note: Analysis functions need to be implemented.[/yellow]")


@cli.command()
def explore():
    """Explore the basic dataset info."""
    console.print("[yellow]Loading basic dataset info...[/yellow]")
    cleaner = DataCleaner()
    cleaner.load_data()
    cleaner.basic_info()

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
    console.print("[dim]Use 'preprocess' for the full, saving pipeline.[/dim]")


@cli.command()
@click.option('--skip-pca', is_flag=True, help='Skip PCA dimensionality reduction')
@click.option('--skip-sparse', is_flag=True, help='Skip sparse matrix creation')
@click.option('--no-save', is_flag=True, help='Do not save processed data')
@click.option('--no-cache', is_flag=True, help='Do not use cached results')
@click.option('--clear-cache', is_flag=True, help='Clear existing cache and run fresh')
@click.option('--memory-limit', type=float, default=16.0, help='Memory limit in GB.')
@click.option('--no-validation', is_flag=True, help='Skip validation steps.')
def preprocess(skip_pca, skip_sparse, no_save, no_cache, clear_cache, memory_limit, no_validation):
    """Run the complete, enhanced preprocessing pipeline."""

    pipeline = PreprocessingPipeline()

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
        # Also clear parquet files if necessary (optional, can be large)
        # files_to_clear.extend(p.name for p in PROCESSED_DATA_DIR.glob("*.parquet"))

        for filename in files_to_clear:
            cache_path = PROCESSED_DATA_DIR / filename
            if cache_path.exists():
                cache_path.unlink()
                console.print(f"[yellow]  ✓ Cleared: {cache_path.name}[/yellow]")

    console.print("[cyan]Starting enhanced preprocessing pipeline...[/cyan]")
    # Call the pipeline with options from CLI
    results = pipeline.run_full_pipeline_with_monitoring(
        create_sparse_matrices=not skip_sparse,
        apply_pca=not skip_pca,
        save_results=not no_save,
        use_cache=not no_cache,
        memory_limit_gb=memory_limit,
        validate_steps=not no_validation
    )

    if results:
        console.print("\n[bold green]🎉 Preprocessing pipeline completed successfully![/bold green]")
        console.print("[dim]You can now run ML algorithms or use 'summary', 'get-dataset', 'validate'.[/dim]")
    else:
        console.print("\n[bold red]❌ Preprocessing pipeline did not complete successfully.[/bold red]")


@cli.command()
def summary():
    """Show preprocessing summary statistics from cache."""

    pipeline = PreprocessingPipeline()
    # Refactor load_processed_data to load_cached_results for clarity
    # It should load 'preprocessing_cache.pkl' which contains metrics

    cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
    ml_datasets_gz_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"

    if cache_file.exists():
        console.print(f"[yellow]Attempting to load full cache: {cache_file}...[/yellow]")
        try:
            with open(cache_file, 'rb') as f:
                cached_results = pickle.load(f)

            if pipeline._validate_cache(cached_results) and 'pipeline_metrics' in cached_results:
                console.print("[green]✓ Loaded and validated full cache. Displaying enhanced summary...[/green]")
                pipeline._display_enhanced_summary(cached_results, cached_results['pipeline_metrics'])
            else:
                console.print("[yellow]Cache invalid or incomplete. Trying ML datasets only...[/yellow]")
                summary() # Recurse to try ML datasets

        except Exception as e:
            console.print(f"[red]Error loading cache: {e}. Trying ML datasets only...[/red]")
            summary() # Recurse to try ML datasets

    elif ml_datasets_gz_path.exists():
        console.print(f"[yellow]Full cache not found. Loading ML datasets from {ml_datasets_gz_path}...[/yellow]")
        try:
            with gzip.open(ml_datasets_gz_path, 'rb') as f:
                ml_datasets = pickle.load(f)

            console.print("[green]✓ Found processed ML datasets (limited summary):[/green]")
            table = Table(title="Available ML Datasets", box=box.ROUNDED)
            table.add_column("Task", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            for task, data in ml_datasets.items():
                 shape_info = f"{data['X'].shape}" if 'X' in data and hasattr(data['X'], 'shape') else f"{len(data.get('transactions', []))} transactions"
                 table.add_row(task.replace('_', ' ').title(), "✓ Ready", shape_info)
            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading ML datasets: {e}[/red]")

    else:
        console.print("[yellow]No processed data or cache found.[/yellow]")
        console.print(Panel(
            "[yellow]💡 Tip: Run 'python analyze.py preprocess' to create comprehensive processed datasets[/yellow]",
            border_style="yellow"
        ))


@cli.command('get-dataset')
@click.argument('task', type=click.Choice(['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']))
def get_dataset(task):
    """Get information about a specific preprocessed dataset."""

    pipeline = PreprocessingPipeline()
    dataset = pipeline.get_dataset_for_task(task) # This now uses the improved loading

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

    else:
        console.print("[yellow]Dataset found, but format is not standard X/y or association.[/yellow]")
        console.print(dataset.keys())
        return

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
        dataset = pipeline.get_dataset_for_task(task_name) # Uses new loading

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


if __name__ == '__main__':
    cli()