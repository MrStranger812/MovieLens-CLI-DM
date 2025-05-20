# MovieLens-CLI-DM
Comprehensive data science project on the MovieLens 20M dataset: implementing regression, classification, clustering, recommender systems, and association rule mining with detailed analysis and visualizations. Complete with data preprocessing and thorough evaluation metrics.

## 🛠️ Technology Stack

- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **CLI Interface**: Click, Rich, Typer
- **Performance Optimization**: Numba, Cython, Dask
- **Testing**: Pytest, Hypothesis
- **Documentation**: Sphinx, MkDocs## 📊 Evaluation & Visualization

### Evaluation Metrics

Each analytical approach is evaluated using appropriate metrics:

- **Regression**: RMSE, MAE, R², Adjusted R²
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Clustering**: Silhouette score, Davies-Bouldin index, Inertia
- **Recommender Systems**: RMSE, MAE, Precision@K, Recall@K, Diversity, Coverage
- **Association Rules**: Support, Confidence, Lift, Conviction

### Visualization Techniques

Results are visualized through:

- **Statistical Plots**: Histograms, box plots, density plots of ratings and features
- **Correlation Matrices**: Heatmaps showing relationships between variables
- **Learning Curves**: Training and validation performance across iterations
- **Clustering Visualizations**: 2D/3D projections of clusters using t-SNE or UMAP
- **Decision Boundaries**: Visualizing classification model decisions
- **Recommendation Graphs**: Network visualizations of user-item relationships
- **Association Networks**: Graphs showing relationships between frequently co-occurring movies# MovieLens Multi-Analytics Project

A comprehensive data science project analyzing the MovieLens 20M dataset through multiple machine learning techniques: regression, classification, clustering, recommender systems, and association rule mining.

![MovieLens Analytics](https://img.shields.io/badge/MovieLens-Analytics-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data Science](https://img.shields.io/badge/Data%20Science-Project-purple)

## 📝 Overview

This project explores the rich MovieLens 20M dataset through multiple analytical lenses, implementing various machine learning algorithms to extract insights and patterns. From predicting user ratings to clustering similar movies and building recommendation systems, this project demonstrates the application of diverse data science techniques on a large-scale, real-world dataset.

### Key Features

- **Recommender Systems**: Both user-based and item-based collaborative filtering approaches
- **Regression Analysis**: Predicting movie ratings based on various features
- **Classification Models**: Categorizing movies and user preferences
- **Clustering Techniques**: Discovering natural groupings in the movie and user data
- **Association Rule Mining**: Uncovering relationships between movie selections
- **Comprehensive Visualizations**: Rich visual representations of all analysis results
- **Interactive CLI**: Beautiful command-line interface with color-coding and progress indicators

## 🎬 Dataset Information

This project uses the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) collected and maintained by the GroupLens Research group at the University of Minnesota. The dataset consists of:

- **20 million ratings** applied to 27,000+ movies by 138,000+ users
- **465,000+ tag applications** describing movie content
- **Ratings scale** from 0.5 to 5 stars, in half-star increments
- **Data collection period** spanning January 1995 to March 2015
- **Rich metadata** including movie titles, genres, and release dates

The dataset was chosen for its comprehensive coverage, high data quality, and widespread use in academic research on recommender systems. This makes it ideal for benchmarking our implementations against established methods in the literature.

### Data Files

- **`ratings.csv`** (21M+ rows): userId, movieId, rating, timestamp
- **`movies.csv`** (27K+ rows): movieId, title, genres
- **`tags.csv`** (465K+ rows): userId, movieId, tag, timestamp
- **`genome-scores.csv`** (11M+ rows): movieId, tagId, relevance
- **`genome-tags.csv`** (1K+ rows): tagId, tag
- **`links.csv`** (27K+ rows): movieId, imdbId, tmdbId

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/movielens-multi-analytics.git
cd movielens-multi-analytics

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and prepare the dataset
python scripts/prepare_data.py
```

## 🖥️ Usage

### Basic Commands

```bash
# Run the main analysis pipeline
python analyze.py --all

# Run specific analysis
python analyze.py --regression
python analyze.py --classification
python analyze.py --clustering
python analyze.py --recommender
python analyze.py --association

# Get recommendations based on a user ID
python recommend.py --user 42 --count 10

# Visualize clustering results
python visualize.py --clustering

# Export analysis results to reports folder
python analyze.py --all --export
```

### Advanced Usage

```bash
# Tune model hyperparameters
python analyze.py --regression --tune

# Compare different algorithms
python compare.py --algorithms regression --metrics rmse,mae,r2

# Export visualizations to specific format
python visualize.py --all --format png --dpi 300

# Run performance benchmarks
python benchmark.py --algorithms all --metrics time,memory
```

## 📁 Project Structure

```
movielens-multi-analytics/
├── data/                      # Dataset storage
│   ├── raw/                   # Original dataset files
│   └── processed/             # Processed data files
├── movielens/                 # Main package
│   ├── __init__.py
│   ├── cli.py                 # CLI interface code
│   ├── preprocessing/         # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── cleaner.py         # Data cleaning utilities
│   │   └── transformer.py     # Feature engineering
│   ├── models/                # ML model implementations
│   │   ├── __init__.py
│   │   ├── regression.py      # Rating prediction models
│   │   ├── classification.py  # Movie/user classification
│   │   ├── clustering.py      # Movie/user clustering
│   │   ├── collaborative/     # Recommender algorithms
│   │   │   ├── __init__.py
│   │   │   ├── user_based.py  # User-based CF
│   │   │   └── item_based.py  # Item-based CF
│   │   └── association.py     # Association rule mining
│   ├── visualization/         # Visualization utilities
│   │   ├── __init__.py
│   │   ├── plots.py           # Plotting functions
│   │   └── cli_visual.py      # CLI visualization helpers
│   ├── evaluation/            # Evaluation frameworks
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── cross_validation.py # CV implementation
│   └── config.py              # Configuration parameters
├── scripts/                   # Utility scripts
│   ├── prepare_data.py        # Data preparation script
│   └── generate_features.py   # Feature generation
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_regression_analysis.ipynb
│   ├── 03_classification_models.ipynb
│   ├── 04_clustering_analysis.ipynb
│   ├── 05_recommender_systems.ipynb
│   └── 06_association_rules.ipynb
├── tests/                     # Unit and integration tests
├── reports/                   # Generated analysis reports and figures
├── README.md                  # Project documentation
├── requirements.txt           # Package dependencies
├── analyze.py                 # Main analysis entry point
├── recommend.py               # Recommender system entry point
├── visualize.py               # Visualization script
└── benchmark.py               # Performance benchmarking script
```

## 🔍 Analytical Approaches

### 1. Data Preprocessing

- **Cleaning**: Handling missing values, removing duplicates, correcting inconsistencies
- **Feature Engineering**: Creating derived features, encoding categorical variables
- **Normalization**: Scaling numerical features, normalizing user ratings
- **Dimensionality Reduction**: Applying PCA for feature selection where appropriate

### 2. Regression Analysis

Predicting movie ratings based on various features:
- **Linear Regression**: Baseline model for rating prediction
- **Regularized Models**: Ridge and Lasso regression to prevent overfitting
- **Random Forest Regression**: Capturing non-linear relationships
- **Gradient Boosting**: Advanced ensemble method for rating prediction

### 3. Classification Models

Categorizing movies and user preferences:
- **Binary Classification**: Predicting if a user will like a movie (rating ≥ 4)
- **Multi-class Classification**: Predicting rating categories (e.g., low, medium, high)
- **Genre Classification**: Predicting movie genres based on tags and ratings

### 4. Clustering Techniques

Discovering natural groupings in the data:
- **K-means Clustering**: Grouping similar movies based on ratings patterns
- **Hierarchical Clustering**: Building movie similarity dendrograms
- **User Segmentation**: Identifying user groups with similar preferences
- **Genre Clustering**: Finding relationships between movie genres

### 5. Recommender Systems

Multiple recommendation approaches:
- **User-Based Collaborative Filtering**: Finding similar users to make recommendations
- **Item-Based Collaborative Filtering**: Recommending similar movies to ones a user liked
- **Matrix Factorization**: Using SVD for latent factor modeling
- **Hybrid Recommenders**: Combining multiple approaches for better results

### 6. Association Rule Mining

Uncovering relationships between items:
- **Apriori Algorithm**: Finding frequent itemsets in movie selections
- **FP-Growth**: Efficiently discovering frequent patterns
- **Movie Bundles**: Identifying groups of movies commonly watched together
- **Sequential Pattern Mining**: Analyzing temporal watching patterns

## 👥 Team Members

- **[Team Member 1]** - Data preprocessing, CLI interface & Clustering algorithms
- **[Team Member 2]** - Recommender systems & Regression analysis
- **[Team Member 3]** - Classification models & Visualization
- **[Team Member 4]** - Association rule mining & Evaluation frameworks

### Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 guidelines for Python code
- Write meaningful docstrings for all functions, classes, and modules
- Add type hints where appropriate
- Ensure all functions and methods have appropriate error handling
- Write unit tests for all new functionality

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset
- All contributors who have helped shape this project
