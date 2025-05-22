import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from .preprocessing.cleaner import DataCleaner

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

if __name__ == '__main__':
    cli()