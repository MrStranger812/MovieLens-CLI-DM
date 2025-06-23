# MovieLens Multi-Analytics Project

A comprehensive data science project analyzing the MovieLens 20M dataset through multiple machine learning techniques: advanced regression with gradient descent variants, classification, hierarchical and agglomerative clustering, recommender systems, and association rule mining.

![MovieLens Analytics](https://img.shields.io/badge/MovieLens-Analytics-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data Science](https://img.shields.io/badge/Data%20Science-Project-purple)

## 📝 Overview

This project explores the rich MovieLens 20M dataset through multiple analytical lenses, implementing various machine learning algorithms to extract insights and patterns. From predicting user ratings using different gradient descent optimization techniques to discovering movie hierarchies through agglomerative clustering, this project demonstrates the application of diverse data science techniques on a large-scale, real-world dataset.

### Key Features

- **Advanced Regression Analysis**: Implementation and comparison of Batch, Stochastic, and Mini-Batch Gradient Descent
- **Hierarchical Clustering**: Comprehensive agglomerative clustering for movie taxonomy and user segmentation
- **Recommender Systems**: User-based, item-based, and association rule-based recommendations
- **Classification Models**: Binary and multi-class rating prediction, genre prediction, and user type classification
- **Association Rule Mining**: Uncovering temporal and content-based relationships between movie selections
- **Performance Optimization**: GPU-accelerated processing and memory-efficient algorithms for large datasets
- **Comprehensive Visualizations**: Rich visual representations including dendrograms, convergence curves, and cluster maps
- **Interactive CLI**: Beautiful command-line interface with color-coding and progress indicators

## 🎬 Dataset Information

This project uses the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) containing:

- **20 million ratings** applied to 27,000+ movies by 138,000+ users
- **465,000+ tag applications** describing movie content and user preferences
- **Ratings scale** from 0.5 to 5 stars, in half-star increments
- **Data collection period** spanning January 1995 to March 2015

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- CUDA-compatible GPU (optional, for accelerated processing)

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

# For GPU acceleration (optional)
pip install cupy-cuda11x  # Adjust for your CUDA version

# Download and prepare the dataset
python scripts/prepare_data.py
```

## 🖥️ CLI Commands

### Data Preprocessing

Before running any analysis, you need to preprocess the data:

```bash
# Standard preprocessing
python analyze.py preprocess

# GPU-accelerated preprocessing
python analyze.py preprocess --use-gpu

# Fast preprocessing with performance mode
python analyze.py preprocess-fast --performance-mode speed

# Clear cache and run fresh preprocessing
python analyze.py preprocess --clear-cache
```

### Data Exploration

```bash
# Explore basic dataset information
python analyze.py explore

# Show preprocessing summary statistics
python analyze.py summary

# Validate preprocessed datasets
python analyze.py validate

# Get information about specific dataset
python analyze.py get-dataset <task>
# Available tasks: regression, classification, clustering_users, clustering_movies, association_rules
```

### Machine Learning Models

#### 1. Association Rule Mining

Discover patterns in movie viewing behaviors:

```bash
# Run association rule mining with default parameters
python analyze.py association

# Custom parameters
python analyze.py association --min-support 0.01 --min-confidence 0.5 --top-rules 20
```

#### 2. Regression with Gradient Descent

Compare different gradient descent optimization methods:

```bash
# Compare all three gradient descent methods
python analyze.py regression

# Run specific method
python analyze.py regression --method batch --learning-rate 0.01 --iterations 1000

# Use sample for faster testing
python analyze.py regression --sample-size 50000
```

Available methods: `batch`, `sgd`, `mini_batch`, `all`

#### 3. Classification Models

Multiple classification approaches:

```bash
# Run all classification models
python analyze.py classification

# Run specific classification model
python analyze.py classification --model rating --sample-size 10000 --n-jobs 4

# Available models:
# - rating: Binary classification (satisfied/not satisfied)
# - genre: Multi-label genre prediction
# - user_type: User behavioral type classification
# - all: Run all classification models
```

#### 4. Clustering Analysis

##### User Segmentation
```bash
# Run user segmentation with both methods
python analyze.py user-segmentation

# Specific method
python analyze.py user-segmentation --method kmeans --n-clusters 8 --save-plots

# Methods: kmeans, agglomerative, both
```

##### Movie Clustering
```bash
# Hierarchical movie clustering
python analyze.py movie-clustering --linkage ward --n-clusters 20

# Compare all linkage methods
python analyze.py movie-clustering --linkage all --plot-dendrogram

# Linkage methods: ward, complete, average, single, all
```

##### Complete Clustering Analysis
```bash
# Run complete analysis (users + movies)
python analyze.py clustering

# With cross-analysis
python analyze.py clustering --cross-analysis

# Show cluster details
python analyze.py cluster-details --user-cluster 3 --movie-cluster 5 --top-n 20
```

#### 5. Collaborative Filtering

Recommender systems using collaborative filtering:

```bash
# Compare user-based and item-based CF
python analyze.py collaborative

# Run specific method
python analyze.py collaborative --method user --k-neighbors 50

# Methods: user, item, both
```

#### 6. Get Recommendations

Get personalized movie recommendations:

```bash
# Using association rules
python analyze.py recommend --user-id 123 --n-recommendations 10 --method association

# Using collaborative filtering
python analyze.py recommend --user-id 123 --n-recommendations 10 --method collaborative

# Methods: association, collaborative, hybrid (not implemented yet)
```

### Utility Commands

```bash
# List available environments
python analyze.py env list

# Run benchmarks
python analyze.py benchmark --iterations 3 --modes speed,balanced,memory

# Clean data (demonstration only)
python analyze.py clean
```

## 📊 Model Details

### Regression with Gradient Descent

Three optimization methods are implemented and compared:

- **Batch Gradient Descent**: Processes entire dataset per iteration
- **Stochastic Gradient Descent (SGD)**: Updates after each sample
- **Mini-Batch Gradient Descent**: Balanced approach with configurable batch sizes

### Classification Models

- **Rating Classification**: Binary (satisfied/not) and multi-class (5 rating levels)
- **Genre Prediction**: Multi-label classification for movie genres
- **User Type Classification**: Behavioral segmentation (critic, enthusiast, casual viewer)

### Clustering

- **User Segmentation**: K-means and agglomerative clustering based on rating patterns
- **Movie Clustering**: Hierarchical clustering with multiple linkage methods
- **Cross-Analysis**: Interaction patterns between user and movie clusters

### Recommender Systems

- **Association Rules**: FP-Growth and Apriori algorithms for pattern mining
- **User-Based CF**: Neighborhood-based collaborative filtering
- **Item-Based CF**: Item similarity with multiple metrics
- **Hybrid Approaches**: Combining multiple recommendation strategies

## 📈 Expected Outputs

Each model produces:

- Performance metrics (RMSE, MAE, accuracy, etc.)
- Visualizations (saved in `reports/` directory)
- Model artifacts (saved in `data/processed/models/`)
- Detailed console output with progress tracking

## 🔍 Project Structure

```
movielens-multi-analytics/
├── analyze.py              # Main CLI entry point
├── movielens/
│   ├── cli.py             # CLI implementation
│   ├── config.py          # Configuration settings
│   ├── preprocessing/     # Data preprocessing modules
│   ├── models/           # ML model implementations
│   │   ├── association.py
│   │   ├── regression.py
│   │   ├── classification.py
│   │   ├── clustering.py
│   │   └── collaborative/
│   ├── evaluation/       # Evaluation metrics
│   └── visualization/    # Plotting utilities
├── data/
│   ├── raw/             # Original dataset files
│   └── processed/       # Preprocessed data and models
├── reports/             # Generated visualizations
└── requirements.txt     # Python dependencies
```

## 💡 Usage Tips

1. **First Time Setup**: Always run preprocessing before any analysis
2. **Memory Management**: Use `--sample-size` for testing on limited memory
3. **GPU Acceleration**: Install CuPy and cuML for GPU support
4. **Caching**: Preprocessed data is cached; use `--clear-cache` to refresh
5. **Parallel Processing**: Adjust `--n-jobs` based on your CPU cores

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset
- The scikit-learn community for excellent machine learning tools
- NVIDIA for CUDA support enabling GPU-accelerated computations

---

*This project demonstrates comprehensive data science techniques on a real-world recommendation system dataset, providing both theoretical insights and practical implementations.*