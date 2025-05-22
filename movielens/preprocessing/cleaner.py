import pandas as pd
import numpy as np
from typing import Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from ..config import *

class DataCleaner:
    """Data cleaning and preprocessing utilities."""

    ratings_df: pd.DataFrame | None
    movies_df: pd.DataFrame | None
    tags_df: pd.DataFrame | None
    
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        self.console = Console()
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all dataset files with progress bar."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Loading ratings.csv...", total=3)
            try:
                self.ratings_df = pd.read_csv(RATINGS_FILE)
                progress.update(task, advance=1, description="[cyan]Loading movies.csv...")
                self.movies_df = pd.read_csv(MOVIES_FILE)
                progress.update(task, advance=1, description="[cyan]Loading tags.csv...")
                self.tags_df = pd.read_csv(TAGS_FILE)
                progress.update(task, advance=1, description="[green]All files loaded!")
            except FileNotFoundError as e:
                self.console.print(f"[bold red]File not found: {e.filename}[/bold red]")
                raise
        self.console.print(f"Loaded {len(self.ratings_df)} ratings")
        self.console.print(f"Loaded {len(self.movies_df)} movies")
        self.console.print(f"Loaded {len(self.tags_df)} tags")
        return self.ratings_df, self.movies_df, self.tags_df
    
    def validate_data(self) -> bool:
        """Validate loaded data integrity."""
        issues = []
        
        if self.ratings_df is not None:
            # Check rating range
            if not self.ratings_df['rating'].between(0.5, 5.0).all():
                issues.append("Invalid ratings found outside 0.5-5.0 range")
            
            # Check for duplicates
            duplicates = self.ratings_df.duplicated(['userId', 'movieId', 'timestamp']).sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate ratings")
        
        if issues:
            for issue in issues:
                self.console.print(f"[yellow]⚠ Warning: {issue}[/yellow]")
            return False
        else:
            self.console.print("[green]✓ Data validation passed[/green]")
            return True

    def basic_info(self):
        """Display basic information about the datasets."""
        if self.ratings_df is not None:
            self.console.print("\n=== Ratings Dataset ===")
            self.console.print(self.ratings_df.info())
            self.console.print(f"Rating range: {self.ratings_df['rating'].min()} - {self.ratings_df['rating'].max()}")
            self.console.print(f"Unique users: {self.ratings_df['userId'].nunique()}")
            self.console.print(f"Unique movies: {self.ratings_df['movieId'].nunique()}")

        if self.movies_df is not None:
            self.console.print("\n=== Movies Dataset ===")
            self.console.print(self.movies_df.info())

        if self.tags_df is not None:
            self.console.print("\n=== Tags Dataset ===")
            self.console.print(self.tags_df.info())

    def clean_ratings(self, save=True, optimize_memory=True) -> pd.DataFrame:
        """Clean and preprocess ratings data."""
        if self.ratings_df is not None:
            # Convert timestamp to datetime
            self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'])
            
            # Add temporal features
            self.ratings_df['year'] = self.ratings_df['timestamp'].dt.year
            self.ratings_df['month'] = self.ratings_df['timestamp'].dt.month
            self.ratings_df['hour'] = self.ratings_df['timestamp'].dt.hour
            self.ratings_df['day_of_week'] = self.ratings_df['timestamp'].dt.day_name()
            
            # Optimize dtypes if requested
            if optimize_memory:
                self.ratings_df['userId'] = self.ratings_df['userId'].astype('category')
                self.ratings_df['movieId'] = self.ratings_df['movieId'].astype('category')
                self.ratings_df['rating'] = self.ratings_df['rating'].astype('float32')
                # Keep timestamp as datetime64[ns] - don't convert to int64
                self.ratings_df['year'] = self.ratings_df['year'].astype('int16')  # Years don't need category
                self.ratings_df['month'] = self.ratings_df['month'].astype('int8')
                self.ratings_df['hour'] = self.ratings_df['hour'].astype('int8')
                self.ratings_df['day_of_week'] = self.ratings_df['day_of_week'].astype('category')
            
            
            self.console.print("[green]✓ Timestamps converted to datetime[/green]")
            self.console.print("[green]✓ Temporal features extracted[/green]")
            if optimize_memory:
                self.console.print("[green]✓ Data types optimized[/green]")

            # Save cleaned data if requested
            if save:
                # Ensure processed directory exists
                PROCESSED_DATA_DIR.mkdir(exist_ok=True)
                
                # Save to processed directory
                output_file = PROCESSED_DATA_DIR / "ratings_cleaned.csv"
                self.ratings_df.to_csv(output_file, index=False)
                self.console.print(f"[green]✓ Cleaned ratings saved to {output_file}[/green]")

        return self.ratings_df.copy()

    def clean_movies(self, save=True) -> pd.DataFrame:
        """Clean and preprocess movies data."""
        if self.movies_df is not None:
            # Extract year from title
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$')
            self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
            
            # Clean up title (remove year)
            self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
            
            # Split and encode genres
            self.movies_df['genre_list'] = self.movies_df['genres'].str.split('|')

            self.console.print("[green]✓ Movie titles cleaned and year extracted[/green]")
            self.console.print("[green]✓ Genres processed[/green]")

            # Save cleaned data if requested
            if save:
                PROCESSED_DATA_DIR.mkdir(exist_ok=True)
                output_file = PROCESSED_DATA_DIR / "movies_cleaned.csv"
                self.movies_df.to_csv(output_file, index=False)
                self.console.print(f"[green]✓ Cleaned movies saved to {output_file}[/green]")

        return self.movies_df

    def clean_tags(self, save=True) -> pd.DataFrame:
        """Clean and preprocess tags data."""
        if self.tags_df is not None:
            # Convert timestamp to datetime
            self.tags_df['timestamp'] = pd.to_datetime(self.tags_df['timestamp'], unit='s')
            
            # Remove rows with missing tags
            self.tags_df = self.tags_df.dropna(subset=['tag'])
            
            # Clean tag text (lowercase, strip whitespace)
            self.tags_df['tag_clean'] = self.tags_df['tag'].str.lower().str.strip()

            self.console.print("[green]✓ Tags cleaned and timestamps converted[/green]")

            # Save cleaned data if requested
            if save:
                PROCESSED_DATA_DIR.mkdir(exist_ok=True)
                output_file = PROCESSED_DATA_DIR / "tags_cleaned.csv"
                self.tags_df.to_csv(output_file, index=False)
                self.console.print(f"[green]✓ Cleaned tags saved to {output_file}[/green]")

        return self.tags_df

    def clean_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Clean all datasets and save to processed directory with progress bar."""
        self.console.print("[yellow]Starting comprehensive data cleaning...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Cleaning ratings...", total=3)
            cleaned_ratings = self.clean_ratings(save=True)
            progress.update(task, advance=1, description="[cyan]Cleaning movies...")
            cleaned_movies = self.clean_movies(save=True)
            progress.update(task, advance=1, description="[cyan]Cleaning tags...")
            cleaned_tags = self.clean_tags(save=True)
            progress.update(task, advance=1, description="[green]All cleaning done!")
        self.console.print("[bold green]✓ All data cleaning completed![/bold green]")
        self.console.print(f"[dim]Cleaned files saved in: {PROCESSED_DATA_DIR}[/dim]")
        return cleaned_ratings, cleaned_movies, cleaned_tags
    
    def get_cleaning_summary(self) -> dict:
        """Get summary statistics after cleaning."""
        summary = {}
        
        if self.ratings_df is not None:
            summary['ratings'] = {
                'total_records': len(self.ratings_df),
                'unique_users': self.ratings_df['userId'].nunique(),
                'unique_movies': self.ratings_df['movieId'].nunique(),
                'date_range': (
                    self.ratings_df['timestamp'].min().strftime('%Y-%m-%d'), 
                    self.ratings_df['timestamp'].max().strftime('%Y-%m-%d')
                ),
                'memory_usage_mb': round(self.ratings_df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
        
        if self.movies_df is not None:
            summary['movies'] = {
                'total_movies': len(self.movies_df),
                'unique_genres': len(set([genre for genres in self.movies_df['genre_list'] for genre in genres if genres])),
                'year_range': (
                    self.movies_df['year'].min(), 
                    self.movies_df['year'].max()
                )
            }
        
        if self.tags_df is not None:
            summary['tags'] = {
                'total_tags': len(self.tags_df),
                'unique_tags': self.tags_df['tag_clean'].nunique(),
                'unique_users_with_tags': self.tags_df['userId'].nunique()
            }
        
        return summary