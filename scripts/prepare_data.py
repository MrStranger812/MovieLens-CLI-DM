import os
import pandas as pd
import zipfile
import urllib.request
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress

console = Console()

@click.command()
@click.option('--download-path', default='data/raw/', help='Path to download dataset')
def prepare_data(download_path):
    """Download and prepare MovieLens 20M dataset."""
    
    # Create directories
    Path(download_path).mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if os.path.exists(os.path.join(download_path, 'ratings.csv')):
        console.print("[green]Dataset already exists![/green]")
        return
    
    console.print("[yellow]Please download the MovieLens 20M dataset from:[/yellow]")
    console.print("https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset")
    console.print("[yellow]Extract the files to the data/raw/ directory[/yellow]")
    
    # Basic data validation once downloaded
    required_files = ['ratings.csv', 'movies.csv', 'tags.csv', 'genome-scores.csv', 'genome-tags.csv', 'links.csv']
    
    console.print("\n[bold]After downloading, the following files should be in data/raw/:[/bold]")
    for file in required_files:
        console.print(f"  - {file}")

if __name__ == '__main__':
    prepare_data()