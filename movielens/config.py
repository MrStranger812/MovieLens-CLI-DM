"""Central configuration for the MovieLens Multi‑Analytics project.

All hard‑coded paths, filenames and sensible default hyper‑parameters live
here so that every module (CLI, pipelines, models, notebooks) can import
`movielens.config` and stay in sync.

Keeping values in one place makes it trivial to (a) switch datasets or
folders for experiments and (b) tweak model defaults without hunting
through the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

# ───────────────────────────────────────────────────────── Paths & folders ──

BASE_DIR: Path = Path(__file__).resolve().parent.parent  # project root
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = BASE_DIR / "reports"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
CACHE_DIR: Path = PROCESSED_DATA_DIR  # alias used by older code

# make sure the important folders exist when importing (does not create raw)
for _d in (PROCESSED_DATA_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────── Dataset files ──

# Raw MovieLens 25M CSVs (as distributed by GroupLens)
RATINGS_FILE: Path = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE: Path = RAW_DATA_DIR / "movies.csv"
TAGS_FILE: Path = RAW_DATA_DIR / "tags.csv"
GENOME_SCORES_FILE: Path = RAW_DATA_DIR / "genome-scores.csv"
GENOME_TAGS_FILE: Path = RAW_DATA_DIR / "genome-tags.csv"
LINKS_FILE: Path = RAW_DATA_DIR / "links.csv"

# ─────────────────────────────────────────────────────── Processed artefacts ──

# These are created by the preprocessing pipeline and read by every model.
# Having the names here lets you swap in an alternate set (e.g. a stratified
# sample) by just changing these constants.
HYPER_FEATURES_FILE: Path = PROCESSED_DATA_DIR / "hyper_features.pkl.gz"
ML_DATASETS_FILE: Path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
USER_ITEM_MATRIX_FILE: Path = PROCESSED_DATA_DIR / "user_item_matrix.npz"
MOVIE_PCA_FEATURES_FILE: Path = PROCESSED_DATA_DIR / "movie_pca_features.parquet"
USER_PCA_FEATURES_FILE: Path = PROCESSED_DATA_DIR / "user_pca_features.parquet"
PCA_MODELS_FILE: Path = PROCESSED_DATA_DIR / "pca_models.pkl"

# ─────────────────────────────────────────────────── General experiment opts ──

DEFAULT_TEST_SIZE: float = 0.20
DEFAULT_RANDOM_STATE: int = 42

# ───────────────────────────────────────────────── Model‑specific defaults ──

# Collaborative filtering & neighbours
DEFAULT_N_NEIGHBORS: int = 20

# Association‑rule mining (FP‑Growth)
DEFAULT_MIN_SUPPORT: float = 0.01
DEFAULT_MIN_CONFIDENCE: float = 0.30

# Gradient‑descent based regression
REGRESSION = {
    "learning_rate": 0.01,
    "iterations": 1_000,
}

# Classification (rating / genre / user‑type).  Dict lets you override per‑model
CLASSIFICATION = {
    "n_estimators": 200,
    "max_depth": None,
    "n_jobs": os.cpu_count(),
}

# Clustering defaults
CLUSTERING = {
    "n_clusters_users": 8,
    "n_clusters_movies": 12,
    "linkage": "ward",  # for hierarchical movie clustering
}

# ─────────────────────────────────────────────────────── Performance knobs ──

# True = try to use CuDF/cuML & RAPIDS if available; the CLI will warn &
# gracefully fall back when set True but no GPU is found.
ENABLE_GPU_ACCEL: bool = False

# Parallelism: use all cores by default, but let the user cap via env var
CPU_COUNT: int = os.cpu_count() or 4
MAX_PARALLEL_JOBS: int = int(os.getenv("ML_MLENS_N_JOBS", CPU_COUNT))

# Memory guard: upper bound (GB) a pipeline should try not to exceed.
MEMORY_BUDGET_GB: float = float(os.getenv("ML_MLENS_MEM_GB", 16))

# ─────────────────────────────────────────────────────── Viz preferences ──

FIGURE_SIZE: tuple[int, int] = (12, 8)
DPI: int = 300
COLOR_PALETTE: str = "viridis"

# ─────────────────────────────────────────────────────────── Helper funcs ──

def describe() -> str:
    """Return a short text summary of the most useful paths. Handy in notebooks."""
    lines = [
        f"BASE_DIR           : {BASE_DIR}",
        f"RAW_DATA_DIR       : {RAW_DATA_DIR}",
        f"PROCESSED_DATA_DIR : {PROCESSED_DATA_DIR}",
        f"REPORTS_DIR        : {REPORTS_DIR}",
        f"Ratings CSV        : {RATINGS_FILE.exists()}",
        f"Processed datasets : {ML_DATASETS_FILE.exists()} (\u2192 ml_ready_datasets.pkl.gz)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # quick sanity check when run directly
    print(describe())
