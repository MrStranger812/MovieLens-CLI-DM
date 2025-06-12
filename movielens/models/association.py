"""
Association Rule Mining for MovieLens Dataset
Location: movielens/models/association.py

This module implements association rule mining to discover patterns in movie viewing behaviors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pathlib import Path
import pickle
import gzip
from ..config import PROCESSED_DATA_DIR

console = Console()

class AssociationRuleMiner:
    """
    Association Rule Mining for discovering movie watching patterns.
    
    Discovers frequent patterns and associations between watched movies using
    both Apriori and FP-Growth algorithms.
    """
    
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5):
        """
        Initialize the association rule miner.
        
        Args:
            min_support: Minimum support threshold for frequent itemsets
            min_confidence: Minimum confidence threshold for rules
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.console = Console()
        self.frequent_itemsets_ap = None
        self.frequent_itemsets_fp = None
        self.rules_ap = None
        self.rules_fp = None
        self.movie_map = None
        self.transactions = None
        
    def prepare_transactions(self, ratings_df: pd.DataFrame, 
                           movies_df: pd.DataFrame,
                           rating_threshold: float = 4.0) -> List[List[str]]:
        """
        Prepare transactional data from ratings.
        
        Args:
            ratings_df: DataFrame with user ratings
            movies_df: DataFrame with movie information
            rating_threshold: Minimum rating to consider a movie as "liked"
            
        Returns:
            List of transactions (each transaction is a list of movie titles)
        """
        self.console.print("[cyan]Preparing transactional data...[/cyan]")
        
        # Filter for high ratings only
        high_ratings = ratings_df[ratings_df['rating'] >= rating_threshold]
        
        # Create movie ID to title mapping
        self.movie_map = dict(zip(movies_df['movieId'], movies_df['title']))
        
        # Group movies by user
        user_transactions = high_ratings.groupby('userId')['movieId'].apply(list)
        
        # Convert movie IDs to titles for better interpretability
        transactions = []
        for user_movies in user_transactions:
            movie_titles = [self.movie_map.get(mid, f"Movie_{mid}") for mid in user_movies]
            if len(movie_titles) >= 2:  # Only keep users with at least 2 movies
                transactions.append(movie_titles)
        
        self.transactions = transactions
        self.console.print(f"[green]✓ Created {len(transactions)} transactions[/green]")
        self.console.print(f"[green]✓ Average movies per user: {np.mean([len(t) for t in transactions]):.1f}[/green]")
        
        return transactions
    
    def fit_apriori(self, transactions: Optional[List[List[str]]] = None):
        """
        Apply Apriori algorithm to find frequent itemsets and rules.
        
        Args:
            transactions: Optional transactions data (uses stored if not provided)
        """
        if transactions is None:
            transactions = self.transactions
            
        if transactions is None:
            raise ValueError("No transactions data available. Run prepare_transactions first.")
            
        self.console.print("[cyan]Running Apriori algorithm...[/cyan]")
        
        # Transform transactions for mlxtend
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_array, columns=te.columns_)
        
        # Apply Apriori
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Finding frequent itemsets...", total=None)
            
            self.frequent_itemsets_ap = apriori(
                df_trans, 
                min_support=self.min_support, 
                use_colnames=True,
                verbose=0
            )
            
            progress.update(task, description="[cyan]Generating association rules...")
            
            # Generate rules
            if len(self.frequent_itemsets_ap) > 0:
                self.rules_ap = association_rules(
                    self.frequent_itemsets_ap, 
                    metric="lift", 
                    min_threshold=1.0,
                    support_only=False
                )
                
                # Filter by confidence
                self.rules_ap = self.rules_ap[self.rules_ap['confidence'] >= self.min_confidence]
                
                self.console.print(f"[green]✓ Found {len(self.frequent_itemsets_ap)} frequent itemsets[/green]")
                self.console.print(f"[green]✓ Generated {len(self.rules_ap)} rules[/green]")
            else:
                self.console.print("[yellow]⚠ No frequent itemsets found. Try lowering min_support.[/yellow]")
    
    def fit_fpgrowth(self, transactions: Optional[List[List[str]]] = None):
        """
        Apply FP-Growth algorithm to find frequent itemsets and rules.
        
        Args:
            transactions: Optional transactions data (uses stored if not provided)
        """
        if transactions is None:
            transactions = self.transactions
            
        if transactions is None:
            raise ValueError("No transactions data available. Run prepare_transactions first.")
            
        self.console.print("[cyan]Running FP-Growth algorithm...[/cyan]")
        
        # Transform transactions for mlxtend
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_array, columns=te.columns_)
        
        # Apply FP-Growth
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Finding frequent itemsets...", total=None)
            
            self.frequent_itemsets_fp = fpgrowth(
                df_trans, 
                min_support=self.min_support, 
                use_colnames=True,
                verbose=0
            )
            
            progress.update(task, description="[cyan]Generating association rules...")
            
            # Generate rules
            if len(self.frequent_itemsets_fp) > 0:
                self.rules_fp = association_rules(
                    self.frequent_itemsets_fp, 
                    metric="lift", 
                    min_threshold=1.0
                )
                
                # Filter by confidence
                self.rules_fp = self.rules_fp[self.rules_fp['confidence'] >= self.min_confidence]
                
                self.console.print(f"[green]✓ Found {len(self.frequent_itemsets_fp)} frequent itemsets[/green]")
                self.console.print(f"[green]✓ Generated {len(self.rules_fp)} rules[/green]")
            else:
                self.console.print("[yellow]⚠ No frequent itemsets found. Try lowering min_support.[/yellow]")
    
    def get_top_rules(self, algorithm: str = 'fpgrowth', 
                      metric: str = 'lift', 
                      n: int = 10) -> pd.DataFrame:
        """
        Get top association rules by a specific metric.
        
        Args:
            algorithm: 'apriori' or 'fpgrowth'
            metric: Metric to sort by ('lift', 'confidence', 'support')
            n: Number of top rules to return
            
        Returns:
            DataFrame with top rules
        """
        rules = self.rules_ap if algorithm == 'apriori' else self.rules_fp
        
        if rules is None or len(rules) == 0:
            self.console.print(f"[yellow]No rules found for {algorithm}[/yellow]")
            return pd.DataFrame()
        
        # Sort by metric
        top_rules = rules.nlargest(n, metric)
        
        # Format for display
        display_rules = pd.DataFrame({
            'Antecedent': top_rules['antecedents'].apply(lambda x: ', '.join(list(x)[:2]) + (f' +{len(x)-2}' if len(x) > 2 else '')),
            'Consequent': top_rules['consequents'].apply(lambda x: ', '.join(list(x)[:2]) + (f' +{len(x)-2}' if len(x) > 2 else '')),
            'Support': top_rules['support'],
            'Confidence': top_rules['confidence'],
            'Lift': top_rules['lift']
        })
        
        return display_rules
    
    def get_movie_bundles(self, min_size: int = 2, 
                         max_size: int = 5, 
                         n: int = 10) -> pd.DataFrame:
        """
        Get frequent movie bundles (itemsets with multiple movies).
        
        Args:
            min_size: Minimum number of movies in a bundle
            max_size: Maximum number of movies in a bundle
            n: Number of top bundles to return
            
        Returns:
            DataFrame with movie bundles
        """
        itemsets = self.frequent_itemsets_fp if self.frequent_itemsets_fp is not None else self.frequent_itemsets_ap
        
        if itemsets is None:
            self.console.print("[yellow]No frequent itemsets found. Run fit_apriori or fit_fpgrowth first.[/yellow]")
            return pd.DataFrame()
        
        # Filter by size
        bundles = itemsets[
            itemsets['itemsets'].apply(lambda x: min_size <= len(x) <= max_size)
        ].copy()
        
        if len(bundles) == 0:
            self.console.print(f"[yellow]No bundles found with size {min_size}-{max_size}[/yellow]")
            return pd.DataFrame()
        
        # Sort by support
        bundles = bundles.nlargest(n, 'support')
        
        # Format for display
        bundles['movies'] = bundles['itemsets'].apply(
            lambda x: ', '.join(list(x)[:3]) + (f' +{len(x)-3} more' if len(x) > 3 else '')
        )
        bundles['size'] = bundles['itemsets'].apply(len)
        
        return bundles[['movies', 'size', 'support']]
    
    def get_recommendations(self, watched_movies: List[str], 
                           n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Get movie recommendations based on association rules.
        
        Args:
            watched_movies: List of movie titles the user has watched
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_title, confidence_score) tuples
        """
        rules = self.rules_fp if self.rules_fp is not None else self.rules_ap
        
        if rules is None:
            self.console.print("[yellow]No rules available. Run fit_apriori or fit_fpgrowth first.[/yellow]")
            return []
        
        recommendations = {}
        
        # Find rules where antecedents are subset of watched movies
        watched_set = set(watched_movies)
        
        for _, rule in rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            # Check if user has watched all movies in antecedents
            if antecedents.issubset(watched_set):
                # Recommend movies in consequents that user hasn't watched
                new_movies = consequents - watched_set
                
                for movie in new_movies:
                    if movie not in recommendations:
                        recommendations[movie] = rule['confidence']
                    else:
                        # Take maximum confidence if movie appears in multiple rules
                        recommendations[movie] = max(recommendations[movie], rule['confidence'])
        
        # Sort by confidence and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def display_summary(self):
        """Display summary of mining results."""
        # Create summary table
        summary_table = Table(title="Association Rule Mining Summary", box="rounded")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Apriori", justify="right")
        summary_table.add_column("FP-Growth", justify="right")
        
        # Add metrics
        metrics = [
            ("Transactions", len(self.transactions) if self.transactions else 0, 
             len(self.transactions) if self.transactions else 0),
            ("Frequent Itemsets", 
             len(self.frequent_itemsets_ap) if self.frequent_itemsets_ap is not None else 0,
             len(self.frequent_itemsets_fp) if self.frequent_itemsets_fp is not None else 0),
            ("Association Rules",
             len(self.rules_ap) if self.rules_ap is not None else 0,
             len(self.rules_fp) if self.rules_fp is not None else 0),
            ("Min Support", f"{self.min_support:.3f}", f"{self.min_support:.3f}"),
            ("Min Confidence", f"{self.min_confidence:.2f}", f"{self.min_confidence:.2f}")
        ]
        
        for metric, ap_value, fp_value in metrics:
            summary_table.add_row(metric, str(ap_value), str(fp_value))
        
        self.console.print(summary_table)
        
        # Display top rules
        if self.rules_fp is not None and len(self.rules_fp) > 0:
            self.console.print("\n[bold]Top Association Rules (FP-Growth):[/bold]")
            top_rules = self.get_top_rules('fpgrowth', n=5)
            self.console.print(top_rules)
        
        # Display top bundles
        bundles = self.get_movie_bundles(n=5)
        if not bundles.empty:
            self.console.print("\n[bold]Top Movie Bundles:[/bold]")
            self.console.print(bundles)
    
    def save_results(self, filepath: Optional[Path] = None):
        """Save mining results to file."""
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "association_rules_results.pkl.gz"
        
        results = {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'transactions': self.transactions,
            'movie_map': self.movie_map,
            'frequent_itemsets_ap': self.frequent_itemsets_ap,
            'frequent_itemsets_fp': self.frequent_itemsets_fp,
            'rules_ap': self.rules_ap,
            'rules_fp': self.rules_fp
        }
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        
        self.console.print(f"[green]✓ Results saved to {filepath}[/green]")
    
    def load_results(self, filepath: Optional[Path] = None):
        """Load mining results from file."""
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "association_rules_results.pkl.gz"
        
        if not filepath.exists():
            self.console.print(f"[red]Results file not found: {filepath}[/red]")
            return False
        
        with gzip.open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.min_support = results['min_support']
        self.min_confidence = results['min_confidence']
        self.transactions = results['transactions']
        self.movie_map = results['movie_map']
        self.frequent_itemsets_ap = results['frequent_itemsets_ap']
        self.frequent_itemsets_fp = results['frequent_itemsets_fp']
        self.rules_ap = results['rules_ap']
        self.rules_fp = results['rules_fp']
        
        self.console.print(f"[green]✓ Results loaded from {filepath}[/green]")
        return True


def run_association_mining_pipeline(ratings_df: pd.DataFrame, 
                                   movies_df: pd.DataFrame,
                                   min_support: float = 0.01,
                                   min_confidence: float = 0.5) -> AssociationRuleMiner:
    """
    Run complete association rule mining pipeline.
    
    Args:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie information
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        Fitted AssociationRuleMiner object
    """
    console.print("[bold cyan]Starting Association Rule Mining Pipeline[/bold cyan]")
    
    # Initialize miner
    miner = AssociationRuleMiner(min_support, min_confidence)
    
    # Prepare transactions
    transactions = miner.prepare_transactions(ratings_df, movies_df)
    
    # Run both algorithms
    miner.fit_fpgrowth(transactions)  # FP-Growth is usually faster
    miner.fit_apriori(transactions)
    
    # Display summary
    miner.display_summary()
    
    # Save results
    miner.save_results()
    
    return miner


if __name__ == "__main__":
    # Example usage
    console.print("[yellow]This module should be imported and used through the main pipeline[/yellow]")
    console.print("Example usage:")
    console.print("from movielens.models.association import run_association_mining_pipeline")
    console.print("miner = run_association_mining_pipeline(ratings_df, movies_df)")