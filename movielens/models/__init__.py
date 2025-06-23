# __init__.py for models

from .association import AssociationRuleMiner, run_association_mining_pipeline
from .classification import (
    RatingClassifier,
    GenrePredictor,
    UserTypeClassifier,
    run_classification_pipeline
)
from .clustering import (
    UserSegmentation,
    HierarchicalMovieClustering,
    AgglomerativeComparison,
    run_user_segmentation_pipeline,
    run_movie_clustering_pipeline,
    run_complete_clustering_analysis
)

from .regression import (
    HyperOptimizedGradientDescent,
    HybridRatingPredictionPipeline, 
    run_optimized_regression_pipeline,
    train_single_gradient_descent_method
)

from .collaborative import *
