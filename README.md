# MovieLens Multi-Analytics Project

A comprehensive data science project analyzing the MovieLens 20M dataset through multiple machine learning techniques: advanced regression with gradient descent variants, classification, hierarchical and agglomerative clustering, recommender systems, and association rule mining.

![MovieLens Analytics](https://img.shields.io/badge/MovieLens-Analytics-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data Science](https://img.shields.io/badge/Data%20Science-Project-purple)

## 📝 Overview

This project explores the rich MovieLens 20M dataset through multiple analytical lenses, implementing various machine learning algorithms to extract insights and patterns. From predicting user ratings using different gradient descent optimization techniques to discovering movie hierarchies through agglomerative clustering, this project demonstrates the application of diverse data science techniques on a large-scale, real-world dataset.

The project goes beyond standard implementations by focusing on algorithmic comparisons and optimization techniques particularly relevant to large-scale recommendation systems, making it suitable for both educational purposes and research applications.

### Key Features

- **Advanced Regression Analysis**: Implementation and comparison of Batch, Stochastic, and Mini-Batch Gradient Descent
- **Hierarchical Clustering**: Comprehensive agglomerative clustering for movie taxonomy and user segmentation
- **Recommender Systems**: Both user-based and item-based collaborative filtering approaches with matrix factorization
- **Classification Models**: Multi-class categorization of movies and user preference prediction
- **Association Rule Mining**: Uncovering temporal and content-based relationships between movie selections
- **Performance Optimization**: GPU-accelerated processing and memory-efficient algorithms for large datasets
- **Comprehensive Visualizations**: Rich visual representations including dendrograms, convergence curves, and cluster maps
- **Interactive CLI**: Beautiful command-line interface with color-coding and progress indicators

## 🎬 Dataset Information

This project uses the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) collected and maintained by the GroupLens Research group at the University of Minnesota. The dataset consists of:

- **20 million ratings** applied to 27,000+ movies by 138,000+ users
- **465,000+ tag applications** describing movie content and user preferences
- **Ratings scale** from 0.5 to 5 stars, in half-star increments
- **Data collection period** spanning January 1995 to March 2015
- **Rich metadata** including movie titles, genres, release dates, and user demographics

The dataset was chosen for its comprehensive coverage, high data quality, and widespread use in academic research on recommender systems. The scale of the dataset (20M+ samples) makes it ideal for demonstrating the practical differences between optimization algorithms that are often only shown on toy datasets.

### Data Files

- **`ratings.csv`** (21M+ rows): userId, movieId, rating, timestamp
- **`movies.csv`** (27K+ rows): movieId, title, genres
- **`tags.csv`** (465K+ rows): userId, movieId, tag, timestamp
- **`genome-scores.csv`** (11M+ rows): movieId, tagId, relevance
- **`genome-tags.csv`** (1K+ rows): tagId, tag
- **`links.csv`** (27K+ rows): movieId, imdbId, tmdbId

## 🛠️ Technology Stack

- **Data Processing**: Pandas, NumPy, SciPy, Dask (for large-scale processing)
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, PyTorch
- **Optimization**: Custom gradient descent implementations, CuPy (GPU acceleration)
- **Clustering**: Scikit-learn, SciPy (hierarchical methods), custom agglomerative algorithms
- **Visualization**: Matplotlib, Seaborn, Plotly, NetworkX (for dendrograms and networks)
- **CLI Interface**: Click, Rich, Typer
- **Performance Optimization**: Numba, Cython, multiprocessing
- **Testing**: Pytest, Hypothesis
- **Documentation**: Sphinx, MkDocs

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

## 🖥️ Usage

### Basic Commands

```bash
# Run the complete analysis pipeline
python analyze.py --all

# Run specific analysis modules
python analyze.py --regression --gradient-descent-comparison
python analyze.py --classification
python analyze.py --clustering --agglomerative
python analyze.py --recommender
python analyze.py --association

# Compare gradient descent methods
python analyze.py --regression --compare-gd --methods batch,sgd,mini_batch

# Generate agglomerative clustering dendrograms
python analyze.py --clustering --agglomerative --dendrogram

# Get recommendations based on user ID
python recommend.py --user 42 --count 10

# Visualize clustering results
python visualize.py --clustering --method agglomerative

# Export analysis results to reports folder
python analyze.py --all --export
```

### Advanced Usage

```bash
# Hyperparameter tuning for gradient descent
python analyze.py --regression --tune-gd --learning-rates 0.001,0.01,0.1 --batch-sizes 1000,5000,10000

# Compare clustering linkage methods
python analyze.py --clustering --agglomerative --linkage ward,complete,average --compare

# Performance benchmarking
python benchmark.py --algorithms gradient_descent,agglomerative --metrics time,memory,accuracy

# GPU-accelerated processing
python analyze.py --all --gpu --batch-size 50000

# Real-time recommendation system
python recommend.py --stream --sgd-updates
```

## 🔍 Analytical Approaches

### 1. Data Preprocessing & Feature Engineering

- **Data Cleaning**: Handling missing values, removing duplicates, timestamp normalization
- **Feature Engineering**: 
  - User behavioral features (rating frequency, genre preferences, temporal patterns)
  - Movie content features (genre combinations, popularity metrics, age factors)
  - Interaction features (user-movie affinity, collaborative signals)
- **Scalable Processing**: Memory-efficient processing for 20M+ records using Dask and chunking
- **Normalization**: User rating bias correction, temporal trend adjustment

### 2. Advanced Regression Analysis with Gradient Descent Variants

Our regression module implements and compares three fundamental optimization approaches, specifically designed for large-scale recommendation data:

#### **Batch Gradient Descent**
- **Implementation**: Processes entire 20M rating dataset per iteration
- **Characteristics**: Most stable convergence, highest memory requirements
- **Use Cases**: Final model training, when computational resources are abundant
- **MovieLens Application**: Optimal for offline model training with complete rating history

```python
# Processes all 20M ratings simultaneously
# Memory: ~8GB RAM required
# Convergence: Smooth, predictable
# Training Time: Slowest but most stable
```

#### **Stochastic Gradient Descent (SGD)**
- **Implementation**: Updates model parameters after each individual rating
- **Characteristics**: Fastest initial convergence, most memory efficient
- **Use Cases**: Online learning, real-time recommendation updates
- **MovieLens Application**: Ideal for streaming new ratings as users interact with the system

```python
# Processes one rating at a time
# Memory: ~100MB RAM required
# Convergence: Fast but noisy
# Training Time: Fastest, suitable for real-time systems
```

#### **Mini-Batch Gradient Descent**
- **Implementation**: Processes ratings in configurable batches (1K-50K samples)
- **Characteristics**: Optimal balance of speed, stability, and memory usage
- **Use Cases**: Production recommendation systems, GPU-accelerated training
- **MovieLens Application**: Best overall choice for most recommendation scenarios

```python
# Configurable batch sizes: 1,000 to 50,000 ratings
# Memory: 500MB-2GB RAM depending on batch size
# Convergence: Stable with good speed
# Training Time: Balanced, GPU-friendly
```

#### **Comparative Analysis Framework**
Our implementation provides comprehensive comparison across multiple dimensions:

- **Convergence Analysis**: Loss curves, epochs to convergence, stability metrics
- **Performance Metrics**: RMSE, MAE, R², prediction accuracy on test sets
- **Computational Efficiency**: Training time, memory usage, scalability analysis
- **Practical Considerations**: Online learning capability, hyperparameter sensitivity

#### **Feature Engineering for Gradient Descent**
Optimized feature sets designed for recommendation systems:

```python
features = {
    'user_features': [
        'user_avg_rating',      # Historical rating average
        'user_rating_std',      # Rating variance (strict vs lenient)
        'user_activity_level',  # Number of ratings given
        'user_genre_preferences', # Weighted genre affinity scores
        'user_temporal_patterns'  # Time-based viewing habits
    ],
    'movie_features': [
        'movie_avg_rating',     # Overall movie rating
        'movie_popularity',     # Number of ratings received
        'movie_age',           # Years since release
        'movie_genre_vector',   # Multi-hot genre encoding
        'movie_tag_similarity'  # Content-based features
    ],
    'interaction_features': [
        'user_movie_genre_affinity', # User preference for movie's genres
        'temporal_rating_context',   # Time-based rating patterns
        'collaborative_signals'      # Matrix factorization features
    ]
}
```

### 3. Hierarchical and Agglomerative Clustering

Our clustering module provides comprehensive hierarchical analysis specifically designed for recommendation system insights:

#### **Agglomerative Movie Clustering**
- **Objective**: Discover natural movie taxonomies beyond traditional genre classifications
- **Linkage Methods**: Ward, Complete, Average, and Single linkage with comparison framework
- **Distance Metrics**: Cosine similarity for sparse rating vectors, Euclidean for dense features
- **Applications**: 
  - Movie recommendation through cluster similarity
  - Content discovery and taxonomy creation
  - Genre relationship analysis

```python
# Movie clustering based on user rating patterns
# Input: 27K movies × rating pattern vectors
# Output: Hierarchical movie taxonomy with similarity scores
# Insights: Discover sub-genres, cross-genre relationships, user preference clusters
```

#### **User Segmentation through Agglomerative Clustering**
- **Objective**: Identify natural user behavioral segments for personalized recommendations
- **Features**: Rating behavior, genre preferences, activity patterns, temporal habits
- **Segmentation Types**:
  - **Behavioral Segments**: Casual viewers, enthusiasts, critics, genre specialists
  - **Preference Segments**: Action lovers, drama enthusiasts, comedy fans, etc.
  - **Activity Segments**: Binge watchers, casual browsers, new release followers

```python
# User clustering based on comprehensive behavioral features
# Input: 138K users × behavioral feature vectors
# Output: User segments with characteristic profiles
# Applications: Personalized UI, targeted recommendations, marketing segmentation
```

#### **Temporal Movie Evolution Clustering**
- **Objective**: Analyze how movie clusters and genres evolve over time (1995-2015)
- **Method**: Time-window based agglomerative clustering with trend analysis
- **Insights**: 
  - Genre popularity evolution
  - Emergence of new movie categories
  - Prediction of future trends

#### **Genre Hierarchy Discovery**
- **Objective**: Build data-driven genre taxonomies using co-occurrence patterns
- **Method**: Agglomerative clustering of genre combinations in movie metadata
- **Output**: Hierarchical genre relationships with statistical significance

```python
# Example discovered hierarchy:
# Action → [Action-Adventure, Action-Thriller, Action-Comedy]
# Drama → [Romance-Drama, Crime-Drama, Historical-Drama]
# Comedy → [Romantic-Comedy, Action-Comedy, Dark-Comedy]
```

#### **Visualization and Interpretation**
- **Dendrograms**: Interactive hierarchical tree visualizations with cluster statistics
- **Cluster Maps**: 2D/3D projections using t-SNE and UMAP for high-dimensional clusters
- **Similarity Networks**: Graph-based visualizations of movie and user relationships
- **Temporal Evolution**: Animated cluster changes over time periods

### 4. Classification Models

Multi-level classification approaches for user and movie categorization:

#### **Rating Prediction Classification**
- **Binary Classification**: Predicting user satisfaction (rating ≥ 4.0 vs < 4.0)
- **Multi-class Classification**: Categorizing ratings into quintiles (very low to very high)
- **Ordinal Classification**: Respecting the inherent order in rating scales

#### **Content Classification**
- **Genre Prediction**: Predicting movie genres based on user rating patterns and tags
- **User Type Classification**: Categorizing users into behavioral types (critic, casual, enthusiast)
- **Recommendation Context**: Classifying when and why users rate movies highly

#### **Advanced Classification Techniques**
- **Ensemble Methods**: Random Forest, Gradient Boosting, Voting classifiers
- **Deep Learning**: Neural networks for complex pattern recognition
- **Imbalanced Learning**: Handling skewed rating distributions and rare genres

### 5. Recommender Systems

Comprehensive recommendation approaches with clustering and optimization integration:

#### **Collaborative Filtering Enhanced**
- **User-Based CF**: Leveraging user clusters from agglomerative analysis for improved similarity computation
- **Item-Based CF**: Using movie clusters to enhance item similarity calculations
- **Hybrid Clustering-CF**: Combining cluster memberships with traditional CF methods

#### **Matrix Factorization with Gradient Descent**
- **SVD/NMF**: Traditional matrix factorization with gradient descent optimization comparison
- **Neural Collaborative Filtering**: Deep learning approaches with custom gradient descent implementations
- **Temporal Factorization**: Time-aware matrix factorization using mini-batch gradient descent

#### **Cluster-Based Recommendations**
- **Intra-Cluster Recommendations**: Recommending movies within the same cluster as user's preferences
- **Cross-Cluster Discovery**: Suggesting movies from related clusters for diversity
- **Hierarchical Recommendations**: Using dendrogram structure for multi-level recommendations

### 6. Association Rule Mining

Advanced pattern discovery in user behavior and movie relationships:

#### **Temporal Pattern Mining**
- **Sequential Patterns**: Analyzing movie watching sequences over time
- **Seasonal Trends**: Discovering time-based viewing patterns (holidays, weekends)
- **User Journey Analysis**: Understanding how user preferences evolve

#### **Content Association Rules**
- **Genre Co-occurrence**: Finding frequently watched genre combinations
- **Movie Bundles**: Identifying movies commonly rated together
- **Tag-Based Associations**: Discovering relationships through user-generated tags

#### **Advanced Mining Techniques**
- **FP-Growth Algorithm**: Efficient frequent pattern discovery on large datasets
- **Constraint-Based Mining**: Focus on specific user segments or time periods
- **Rare Pattern Discovery**: Finding unusual but significant viewing patterns

## 📊 Evaluation & Visualization

### Comprehensive Evaluation Metrics

#### **Regression Evaluation**
- **Traditional Metrics**: RMSE, MAE, R², Adjusted R²
- **Gradient Descent Specific**: 
  - Convergence rate analysis
  - Training time vs accuracy trade-offs
  - Memory usage efficiency
  - Hyperparameter sensitivity analysis

#### **Clustering Evaluation**
- **Internal Validation**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- **External Validation**: 
  - Genre purity scores
  - User satisfaction through recommendation quality
  - Temporal stability of clusters
- **Interpretability Metrics**: Cluster size distribution, feature importance, cluster characteristics

#### **Recommendation System Evaluation**
- **Accuracy Metrics**: RMSE, MAE for rating prediction
- **Ranking Metrics**: Precision@K, Recall@K, NDCG, MAP
- **Diversity Metrics**: Intra-list diversity, coverage, novelty
- **Business Metrics**: User engagement, satisfaction surveys

### Advanced Visualization Techniques

#### **Gradient Descent Visualizations**
- **Convergence Curves**: Loss function evolution for all three methods
- **Learning Rate Analysis**: 3D surface plots showing convergence landscapes
- **Batch Size Impact**: Performance vs computational cost trade-off curves
- **Real-time Training**: Live updating loss curves during training

#### **Clustering Visualizations**
- **Interactive Dendrograms**: Hierarchical tree structures with cluster statistics
- **Cluster Evolution**: Time-lapse visualizations of cluster changes
- **Feature Space Projections**: t-SNE and UMAP embeddings of high-dimensional clusters
- **Similarity Networks**: Graph visualizations of movie and user relationships

#### **Recommendation Visualizations**
- **User-Item Interaction Maps**: Heatmaps of rating patterns
- **Recommendation Networks**: Graph-based visualization of recommendation flow
- **Cluster-Based Recommendations**: Visual representation of cluster-driven suggestions
- **Performance Dashboards**: Real-time monitoring of recommendation system metrics

## 🎯 Research Questions and Insights

### Gradient Descent Research Questions
1. **Scalability Analysis**: How do different gradient descent methods scale with MovieLens's 20M samples?
2. **Convergence Behavior**: Which optimization method converges fastest on sparse recommendation data?
3. **Online Learning**: Can SGD effectively handle streaming ratings for real-time recommendations?
4. **Feature Engineering Impact**: How do different feature engineering approaches affect gradient descent performance?
5. **Memory-Performance Trade-offs**: What are the optimal batch sizes for different hardware configurations?

### Clustering Research Questions
1. **Movie Taxonomy Discovery**: Can agglomerative clustering reveal unknown genre relationships and sub-categories?
2. **User Behavioral Segmentation**: What natural user segments exist beyond demographic classifications?
3. **Temporal Evolution**: How do movie and user clusters evolve over the 20-year dataset period?
4. **Recommendation Enhancement**: How much do hierarchical clusters improve recommendation quality compared to flat clustering?
5. **Cross-Domain Insights**: Can clustering patterns in MovieLens inform other recommendation domains?

### Integration Research Questions
1. **Method Synergy**: How can gradient descent optimization enhance clustering algorithms for large datasets?
2. **Hierarchical Optimization**: Can cluster structure inform gradient descent initialization and convergence?
3. **Multi-Objective Optimization**: How do accuracy, speed, and interpretability trade off across different method combinations?

## 👥 Team Members

- **[Team Member 1]** - Data preprocessing, CLI interface & Advanced clustering algorithms
- **[Team Member 2]** - Gradient descent implementations & Recommender systems
- **[Team Member 3]** - Classification models & Visualization frameworks
- **[Team Member 4]** - Association rule mining & Evaluation frameworks

### Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 guidelines for Python code
- Write comprehensive docstrings for all functions, classes, and modules
- Add type hints for better code maintainability
- Implement proper error handling and logging
- Write unit tests for all new functionality
- Include performance benchmarks for optimization algorithms

## 📈 Expected Outcomes and Impact

### Academic Contributions
- **Comparative Analysis**: Comprehensive comparison of gradient descent methods on large-scale recommendation data
- **Clustering Insights**: Novel discoveries about movie and user relationship hierarchies
- **Optimization Research**: Practical insights into algorithm selection for recommendation systems
- **Scalability Studies**: Guidelines for algorithm choice based on dataset characteristics

### Practical Applications
- **Production Recommendations**: Ready-to-deploy recommendation system with optimized algorithms
- **User Segmentation**: Actionable user segments for personalized experiences
- **Content Organization**: Data-driven movie taxonomies for improved content discovery
- **Real-time Systems**: Online learning framework for streaming recommendation updates

### Educational Value
- **Algorithm Comparison**: Clear demonstration of trade-offs between different optimization approaches
- **Large-Scale Processing**: Practical experience with big data machine learning techniques
- **Visualization Mastery**: Advanced visualization techniques for complex analytical results
- **Research Methodology**: Comprehensive evaluation framework for machine learning research

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset
- The scikit-learn community for excellent clustering and machine learning tools
- NVIDIA for CUDA support enabling GPU-accelerated computations
- All contributors who have helped shape this project into a comprehensive research platform

---

*This project represents a significant advancement in recommendation system research, providing both theoretical insights and practical implementations that bridge the gap between academic research and industry applications.*
