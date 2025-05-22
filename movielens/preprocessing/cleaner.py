import pandas as pd
import numpy as np
from typing import Tuple
from ..config import *

class DataCleaner:
    """Data cleaning and preprocessing utilities."""
    
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.tags_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all dataset files."""
        self.ratings_df = pd.read_csv(RATINGS_FILE)
        self.movies_df = pd.read_csv(MOVIES_FILE)
        self.tags_df = pd.read_csv(TAGS_FILE)
        
        print(f"Loaded {len(self.ratings_df)} ratings")
        print(f"Loaded {len(self.movies_df)} movies")
        print(f"Loaded {len(self.tags_df)} tags")
        
        return self.ratings_df, self.movies_df, self.tags_df
    
    def basic_info(self):
        """Display basic information about the datasets."""
        if self.ratings_df is not None:
            print("\n=== Ratings Dataset ===")
            print(self.ratings_df.info())
            print(f"Rating range: {self.ratings_df['rating'].min()} - {self.ratings_df['rating'].max()}")
            print(f"Unique users: {self.ratings_df['userId'].nunique()}")
            print(f"Unique movies: {self.ratings_df['movieId'].nunique()}")
            
        if self.movies_df is not None:
            print("\n=== Movies Dataset ===")
            print(self.movies_df.info())
            
        if self.tags_df is not None:
            print("\n=== Tags Dataset ===")
            print(self.tags_df.info())