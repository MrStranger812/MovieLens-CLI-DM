import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset files
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
TAGS_FILE = RAW_DATA_DIR / "tags.csv"
GENOME_SCORES_FILE = RAW_DATA_DIR / "genome-scores.csv"
GENOME_TAGS_FILE = RAW_DATA_DIR / "genome-tags.csv"
LINKS_FILE = RAW_DATA_DIR / "links.csv"

# Model parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_NEIGHBORS = 20
DEFAULT_MIN_SUPPORT = 0.01
DEFAULT_MIN_CONFIDENCE = 0.3

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "viridis"