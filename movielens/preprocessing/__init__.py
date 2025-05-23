# movielens/preprocessing/__init__.py
"""
Data preprocessing and transformation modules.

This package provides comprehensive data preprocessing capabilities including:
- Data cleaning and validation (cleaner.py)
- Feature engineering and transformation (transformer.py)  
- Complete preprocessing pipeline (pipeline.py)
"""

from .cleaner import DataCleaner
from .transformer import DataTransformer
from .pipeline import PreprocessingPipeline

__all__ = ['DataCleaner', 'DataTransformer', 'PreprocessingPipeline']