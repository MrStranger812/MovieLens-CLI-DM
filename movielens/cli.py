import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from .config import PROCESSED_DATA_DIR
from .preprocessing.cleaner import DataCleaner
from .preprocessing.pipeline import PreprocessingPipeline

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
              default='all', help='Analysis technique to run')
def analyze(technique):
    """Run data analysis with specified technique."""
    console.print(f"[green]Starting {technique} analysis...[/green]")
    
    # Load data
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()
    cleaner.basic_info()

@cli.command()
def explore():
    """Explore the dataset."""
    console.print("[yellow]Loading dataset for exploration...[/yellow]")
    
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()
    cleaner.basic_info()

@cli.command()
def clean():
    """Clean and preprocess the data (basic cleaning only)."""
    cleaner = DataCleaner()
    ratings_df, movies_df, tags_df = cleaner.load_data()
    
    # Clean data
    clean_ratings = cleaner.clean_ratings()
    
    console.print("[green]✓ Basic data cleaning completed![/green]")

@cli.command()
@click.option('--skip-pca', is_flag=True, help='Skip PCA dimensionality reduction')
@click.option('--skip-sparse', is_flag=True, help='Skip sparse matrix creation')
@click.option('--no-save', is_flag=True, help='Do not save processed data')
@click.option('--no-cache', is_flag=True, help='Do not use cached results')
@click.option('--clear-cache', is_flag=True, help='Clear existing cache and run fresh')
def preprocess(skip_pca, skip_sparse, no_save, no_cache, clear_cache):
    """Run the complete preprocessing pipeline."""
    
    pipeline = PreprocessingPipeline()
    
    # Handle cache clearing
    if clear_cache:
        cache_file = PROCESSED_DATA_DIR / "preprocessing_cache.pkl"
        matrix_cache = PROCESSED_DATA_DIR / "user_item_matrix.npz"
        mappings_cache = PROCESSED_DATA_DIR / "user_item_mappings.pkl"
        
        for cache_path in [cache_file, matrix_cache, mappings_cache]:
            if cache_path.exists():
                cache_path.unlink()
                console.print(f"[yellow]Cleared cache: {cache_path.name}[/yellow]")
    
    results = pipeline.run_full_pipeline(
        create_sparse_matrices=not skip_sparse,
        apply_pca=not skip_pca,
        save_results=not no_save,
        use_cache=not no_cache
    )
    
    console.print("\n[bold green]🎉 Preprocessing pipeline completed successfully![/bold green]")
    console.print("[dim]You can now run ML algorithms using the processed data.[/dim]")

@cli.command()
def summary():
    """Show preprocessing summary statistics."""
    
    # Try to load existing processed data first
    pipeline = PreprocessingPipeline()
    processed_data = pipeline.load_processed_data()
    
    if not processed_data:
        console.print("[yellow]No processed data found. Running basic data exploration...[/yellow]")
        
        # If no processed data, show basic data info
        cleaner = DataCleaner()
        ratings_df, movies_df, tags_df = cleaner.load_data()
        cleaner.clean_all()
        summary_stats = cleaner.get_cleaning_summary()
        
        # Display summary nicely
        for dataset, stats in summary_stats.items():
            table = Table(title=f"{dataset.title()} Statistics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in stats.items():
                table.add_row(metric.replace('_', ' ').title(), str(value))
            
            console.print(table)
            console.print()
        
        console.print(Panel(
            "[yellow]💡 Tip: Run 'python analyze.py preprocess' to create comprehensive processed datasets[/yellow]",
            border_style="yellow"
        ))
    else:
        console.print("[green]✓ Found processed data![/green]")
        # Show processed data summary
        if 'ml_ready_datasets' in processed_data:
            table = Table(title="Available ML Datasets", box=box.ROUNDED)
            table.add_column("Task", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Description", style="dim")
            
            ml_tasks = {
                'regression': 'Rating prediction with numerical features',
                'classification': 'Like/dislike prediction (binary)',
                'clustering_users': 'User behavior segmentation',
                'clustering_movies': 'Movie similarity grouping',
                'association_rules': 'Movie co-occurrence patterns'
            }
            
            for task, description in ml_tasks.items():
                if task in processed_data['ml_ready_datasets']:
                    table.add_row(task.replace('_', ' ').title(), "✓ Ready", description)
                else:
                    table.add_row(task.replace('_', ' ').title(), "✗ Missing", description)
            
            console.print(table)

@cli.command()
@click.argument('task', type=click.Choice(['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']))
def get_dataset(task):
    """Get information about a specific preprocessed dataset."""
    
    pipeline = PreprocessingPipeline()
    dataset = pipeline.get_dataset_for_task(task)
    
    if not dataset:
        console.print(f"[red]Dataset for {task} not found. Run preprocessing first.[/red]")
        return
    
    console.print(f"\n[bold cyan]Dataset Information: {task.replace('_', ' ').title()}[/bold cyan]")
    
    if 'X' in dataset and hasattr(dataset['X'], 'shape'):
        samples, features = dataset['X'].shape
        
        info_table = Table(box=box.ROUNDED)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Samples", f"{samples:,}")
        info_table.add_row("Features", f"{features}")
        
        if 'y' in dataset:
            if hasattr(dataset['y'], 'nunique'):
                unique_targets = dataset['y'].nunique()
                info_table.add_row("Unique Targets", f"{unique_targets}")
            
            if hasattr(dataset['y'], 'value_counts'):
                target_dist = dataset['y'].value_counts()
                info_table.add_row("Target Distribution", str(dict(target_dist.head())))
        
        if 'feature_names' in dataset:
            feature_sample = dataset['feature_names'][:10]
            if len(dataset['feature_names']) > 10:
                feature_sample.append(f"... and {len(dataset['feature_names']) - 10} more")
            info_table.add_row("Sample Features", ", ".join(feature_sample))
        
        console.print(info_table)
        
    elif task == 'association_rules':
        transactions = dataset.get('transactions', [])
        console.print(f"[green]✓ {len(transactions)} transaction sequences ready for association rule mining[/green]")

@cli.command()
@click.option('--task', type=click.Choice(['regression', 'classification', 'clustering', 'recommender', 'association']), 
              help='Specific ML task to validate data for')
def validate(task):
    """Validate preprocessed data for ML algorithms."""
    
    pipeline = PreprocessingPipeline()
    
    if task:
        # Validate specific task
        if task == 'clustering':
            # Check both user and movie clustering datasets
            user_dataset = pipeline.get_dataset_for_task('clustering_users')
            movie_dataset = pipeline.get_dataset_for_task('clustering_movies')
            
            if user_dataset and 'X' in user_dataset:
                console.print(f"[green]✓ User clustering dataset ready: {user_dataset['X'].shape}[/green]")
            else:
                console.print("[red]✗ User clustering dataset not found[/red]")
                
            if movie_dataset and 'X' in movie_dataset:
                console.print(f"[green]✓ Movie clustering dataset ready: {movie_dataset['X'].shape}[/green]")
            else:
                console.print("[red]✗ Movie clustering dataset not found[/red]")
        
        elif task == 'recommender':
            # Check collaborative filtering data
            processed_data = pipeline.load_processed_data()
            if 'collaborative_filtering' in processed_data:
                console.print("[green]✓ Collaborative filtering matrices ready[/green]")
            else:
                console.print("[red]✗ Collaborative filtering data not found[/red]")
        
        else:
            dataset = pipeline.get_dataset_for_task(task)
            if dataset and 'X' in dataset:
                console.print(f"[green]✓ {task.title()} dataset ready: {dataset['X'].shape}[/green]")
            else:
                console.print(f"[red]✗ {task.title()} dataset not found[/red]")
    else:
        # Validate all tasks
        all_tasks = ['regression', 'classification', 'clustering_users', 'clustering_movies', 'association_rules']
        
        validation_table = Table(title="Data Validation Results", box=box.ROUNDED)
        validation_table.add_column("ML Task", style="cyan")
        validation_table.add_column("Status", style="bold")
        validation_table.add_column("Details", style="dim")
        
        for task_name in all_tasks:
            dataset = pipeline.get_dataset_for_task(task_name)
            
            if not dataset:
                validation_table.add_row(
                    task_name.replace('_', ' ').title(),
                    "[red]✗ Missing[/red]",
                    "Dataset not found"
                )
            elif 'X' in dataset and hasattr(dataset['X'], 'shape'):
                samples, features = dataset['X'].shape
                validation_table.add_row(
                    task_name.replace('_', ' ').title(),
                    "[green]✓ Ready[/green]",
                    f"{samples:,} samples, {features} features"
                )
            elif task_name == 'association_rules' and 'transactions' in dataset:
                transaction_count = len(dataset['transactions'])
                validation_table.add_row(
                    task_name.replace('_', ' ').title(),
                    "[green]✓ Ready[/green]",
                    f"{transaction_count:,} transactions"
                )
            else:
                validation_table.add_row(
                    task_name.replace('_', ' ').title(),
                    "[yellow]⚠ Incomplete[/yellow]",
                    "Unexpected format"
                )
        
        console.print(validation_table)
        
        # Show recommendation
        console.print(Panel(
            "[dim]💡 If any datasets are missing, run:[/dim]\n"
            "[bold cyan]python analyze.py preprocess[/bold cyan]",
            border_style="blue"
        ))

if __name__ == '__main__':
    cli()