"""
Regression Models for Rating Prediction with Gradient Descent Variants
Location: movielens/models/regression.py

This module implements rating prediction using different gradient descent optimization approaches:
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
import warnings
from joblib import Parallel, delayed
import multiprocessing as mp
from ..config import PROCESSED_DATA_DIR, REPORTS_DIR

warnings.filterwarnings('ignore')
console = Console()


class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """
    Custom Gradient Descent Regressor for comparing different optimization methods.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 method: str = 'batch',
                 batch_size: int = 32,
                 regularization: str = 'none',
                 lambda_reg: float = 0.01,
                 early_stopping: bool = True,
                 patience: int = 10,
                 verbose: bool = True,
                 random_state: int = 42):
        """
        Initialize the gradient descent regressor.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Maximum number of iterations
            method: 'batch', 'sgd', or 'mini_batch'
            batch_size: Batch size for mini-batch gradient descent
            regularization: 'none', 'l1', 'l2', or 'elastic'
            lambda_reg: Regularization parameter
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            verbose: Whether to print progress
            random_state: Random seed
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.training_time = 0
        self.n_features = None
        
        np.random.seed(random_state)
    
    def _initialize_parameters(self, n_features: int):
        """Initialize weights and bias."""
        # Xavier initialization
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        self.n_features = n_features
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the loss function."""
        predictions = X.dot(self.weights) + self.bias
        mse_loss = np.mean((predictions - y) ** 2)
        
        # Add regularization
        if self.regularization == 'l2':
            reg_loss = self.lambda_reg * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_loss = self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'elastic':
            reg_loss = self.lambda_reg * (0.5 * np.sum(self.weights ** 2) + 
                                          0.5 * np.sum(np.abs(self.weights)))
        else:
            reg_loss = 0
        
        return mse_loss + reg_loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        predictions = X.dot(self.weights) + self.bias
        
        # Gradients of MSE
        dw = (2 / n_samples) * X.T.dot(predictions - y)
        db = (2 / n_samples) * np.sum(predictions - y)
        
        # Add regularization gradients
        if self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'elastic':
            dw += self.lambda_reg * (self.weights + 0.5 * np.sign(self.weights))
        
        return dw, db
    
    def _batch_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Perform batch gradient descent."""
        for iteration in range(self.n_iterations):
            # Compute gradients
            dw, db = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Early stopping
            if self.early_stopping and iteration > self.patience:
                if all(self.loss_history[-self.patience:][i] <= self.loss_history[-self.patience:][i+1] 
                       for i in range(self.patience-1)):
                    if self.verbose:
                        console.print(f"[yellow]Early stopping at iteration {iteration}[/yellow]")
                    break
    
    def _stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Perform stochastic gradient descent."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        for iteration in range(self.n_iterations):
            # Shuffle data
            np.random.shuffle(indices)
            
            epoch_loss = 0
            for idx in indices:
                # Single sample
                X_i = X[idx:idx+1]
                y_i = y[idx:idx+1]
                
                # Compute gradients
                dw, db = self._compute_gradients(X_i, y_i)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                epoch_loss += self._compute_loss(X_i, y_i)
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
            # Early stopping
            if self.early_stopping and iteration > self.patience:
                if all(self.loss_history[-self.patience:][i] <= self.loss_history[-self.patience:][i+1] 
                       for i in range(self.patience-1)):
                    if self.verbose:
                        console.print(f"[yellow]Early stopping at iteration {iteration}[/yellow]")
                    break
    
    def _mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Perform mini-batch gradient descent."""
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // self.batch_size)
        
        for iteration in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)
                
                # Get batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                epoch_loss += self._compute_loss(X_batch, y_batch) * (end_idx - start_idx)
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
            # Early stopping
            if self.early_stopping and iteration > self.patience:
                if all(self.loss_history[-self.patience:][i] <= self.loss_history[-self.patience:][i+1] 
                       for i in range(self.patience-1)):
                    if self.verbose:
                        console.print(f"[yellow]Early stopping at iteration {iteration}[/yellow]")
                    break
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentRegressor':
        """Fit the model using the specified gradient descent method."""
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        self.loss_history = []
        
        # Start timing
        start_time = time.time()
        
        if self.verbose:
            console.print(f"[cyan]Training with {self.method} gradient descent...[/cyan]")
        
        # Choose method
        if self.method == 'batch':
            self._batch_gradient_descent(X, y)
        elif self.method == 'sgd':
            self._stochastic_gradient_descent(X, y)
        elif self.method == 'mini_batch':
            self._mini_batch_gradient_descent(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.training_time = time.time() - start_time
        
        if self.verbose:
            console.print(f"[green]✓ Training completed in {self.training_time:.2f} seconds[/green]")
            console.print(f"[green]✓ Final loss: {self.loss_history[-1]:.4f}[/green]")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return X.dot(self.weights) + self.bias
    
    def get_convergence_info(self) -> Dict:
        """Get convergence information."""
        return {
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'n_iterations': len(self.loss_history),
            'training_time': self.training_time,
            'converged': len(self.loss_history) < self.n_iterations
        }


class RatingPredictionPipeline:
    """
    Complete pipeline for rating prediction with gradient descent comparison.
    """
    
    def __init__(self, n_jobs: int = -1):
        """Initialize the rating prediction pipeline."""
        self.console = Console()
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_ml_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the preprocessed ML dataset for regression."""
        ml_datasets_path = PROCESSED_DATA_DIR / "ml_ready_datasets.pkl.gz"
        
        if not ml_datasets_path.exists():
            raise FileNotFoundError(
                "ML datasets not found. Run 'python analyze.py preprocess' first."
            )
        
        import gzip
        with gzip.open(ml_datasets_path, 'rb') as f:
            ml_datasets = pickle.load(f)
        
        if 'regression' not in ml_datasets or ml_datasets['regression'] is None:
            raise ValueError("Regression dataset not found in ML datasets")
        
        regression_data = ml_datasets['regression']
        X = regression_data['X']
        y = regression_data['y']
        
        self.console.print(f"[green]✓ Loaded dataset: {X.shape[0]:,} samples, {X.shape[1]} features[/green]")
        
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.2, 
                     scale: bool = True) -> None:
        """Prepare data for training."""
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        self.console.print(f"[green]✓ Data split: {len(self.X_train):,} train, {len(self.X_test):,} test[/green]")
    
    def train_gradient_descent_variant(self, 
                                     method: str,
                                     learning_rate: float = 0.01,
                                     n_iterations: int = 1000,
                                     batch_size: int = 32,
                                     regularization: str = 'l2',
                                     lambda_reg: float = 0.01) -> Dict:
        """Train a single gradient descent variant."""
        self.console.print(f"\n[cyan]Training {method.upper()} Gradient Descent...[/cyan]")
        
        # Create model
        model = GradientDescentRegressor(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            method=method,
            batch_size=batch_size,
            regularization=regularization,
            lambda_reg=lambda_reg,
            verbose=False
        )
        
        # Train with progress
        start_time = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Store results
        results = {
            'method': method,
            'model': model,
            'training_time': training_time,
            'convergence_info': model.get_convergence_info(),
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'loss_history': model.loss_history,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'n_iterations': n_iterations,
                'batch_size': batch_size if method == 'mini_batch' else None,
                'regularization': regularization,
                'lambda_reg': lambda_reg
            }
        }
        
        self.console.print(f"[green]✓ {method.upper()} - RMSE: {test_rmse:.4f}, Time: {training_time:.2f}s[/green]")
        
        return results
    
    def compare_gradient_descent_methods(self, 
                                       learning_rates: List[float] = [0.001, 0.01, 0.1],
                                       batch_sizes: List[int] = [32, 128, 512]) -> None:
        """Compare all three gradient descent methods."""
        self.console.print(Panel.fit(
            "[bold cyan]Gradient Descent Methods Comparison[/bold cyan]\n"
            "Comparing Batch, Stochastic, and Mini-Batch Gradient Descent",
            border_style="cyan"
        ))
        
        # Base configuration
        base_config = {
            'n_iterations': 1000,
            'regularization': 'l2',
            'lambda_reg': 0.01
        }
        
        # Test each method
        methods = ['batch', 'sgd', 'mini_batch']
        
        for method in methods:
            best_result = None
            best_rmse = float('inf')
            
            # Grid search for learning rate
            for lr in learning_rates:
                config = base_config.copy()
                config['learning_rate'] = lr
                
                if method == 'mini_batch':
                    # Also search for best batch size
                    for batch_size in batch_sizes:
                        config['batch_size'] = batch_size
                        result = self.train_gradient_descent_variant(method, **config)
                        
                        if result['metrics']['test_rmse'] < best_rmse:
                            best_rmse = result['metrics']['test_rmse']
                            best_result = result
                else:
                    result = self.train_gradient_descent_variant(method, **config)
                    
                    if result['metrics']['test_rmse'] < best_rmse:
                        best_rmse = result['metrics']['test_rmse']
                        best_result = result
            
            self.models[method] = best_result['model']
            self.results[method] = best_result
    
    def display_comparison_results(self):
        """Display comprehensive comparison results."""
        # Create comparison table
        table = Table(title="Gradient Descent Methods Comparison", box="rounded")
        table.add_column("Method", style="cyan")
        table.add_column("Train RMSE", justify="right")
        table.add_column("Test RMSE", justify="right", style="bold")
        table.add_column("MAE", justify="right")
        table.add_column("R²", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Iterations", justify="right")
        table.add_column("LR", justify="right")
        
        for method, result in self.results.items():
            metrics = result['metrics']
            conv_info = result['convergence_info']
            hyperparams = result['hyperparameters']
            
            table.add_row(
                method.upper(),
                f"{metrics['train_rmse']:.4f}",
                f"{metrics['test_rmse']:.4f}",
                f"{metrics['test_mae']:.4f}",
                f"{metrics['test_r2']:.4f}",
                f"{conv_info['training_time']:.2f}",
                str(conv_info['n_iterations']),
                f"{hyperparams['learning_rate']:.3f}"
            )
        
        self.console.print(table)
        
        # Memory and computational analysis
        analysis_table = Table(title="Computational Analysis", box="rounded")
        analysis_table.add_column("Method", style="cyan")
        analysis_table.add_column("Memory Complexity", style="yellow")
        analysis_table.add_column("Time Complexity", style="yellow")
        analysis_table.add_column("Best Use Case", style="green")
        
        analysis_data = [
            ("Batch", "O(n × d)", "O(n × d × iterations)", "Small datasets, stable convergence"),
            ("SGD", "O(d)", "O(d × iterations × n)", "Online learning, large datasets"),
            ("Mini-Batch", "O(b × d)", "O(b × d × iterations × n/b)", "Balance of speed and stability")
        ]
        
        for method, memory, time, use_case in analysis_data:
            analysis_table.add_row(method, memory, time, use_case)
        
        self.console.print("\n")
        self.console.print(analysis_table)
    
    def plot_convergence_curves(self, save_path: Optional[Path] = None):
        """Plot convergence curves for all methods."""
        plt.figure(figsize=(12, 6))
        
        for method, result in self.results.items():
            loss_history = result['loss_history']
            plt.plot(loss_history, label=f"{method.upper()}", linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Gradient Descent Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / 'gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        self.console.print("[green]✓ Convergence plot saved[/green]")
    
    def plot_performance_comparison(self, save_path: Optional[Path] = None):
        """Plot performance comparison bar charts."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = list(self.results.keys())
        
        # RMSE comparison
        rmse_train = [self.results[m]['metrics']['train_rmse'] for m in methods]
        rmse_test = [self.results[m]['metrics']['test_rmse'] for m in methods]
        
        ax = axes[0, 0]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, rmse_train, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, rmse_test, width, label='Test', alpha=0.8)
        ax.set_xlabel('Method')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training time comparison
        ax = axes[0, 1]
        times = [self.results[m]['convergence_info']['training_time'] for m in methods]
        bars = ax.bar(methods, times, alpha=0.8, color='orange')
        ax.set_xlabel('Method')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s', ha='center', va='bottom')
        
        # R² Score comparison
        ax = axes[1, 0]
        r2_train = [self.results[m]['metrics']['train_r2'] for m in methods]
        r2_test = [self.results[m]['metrics']['test_r2'] for m in methods]
        
        x = np.arange(len(methods))
        ax.bar(x - width/2, r2_train, width, label='Train', alpha=0.8, color='green')
        ax.bar(x + width/2, r2_test, width, label='Test', alpha=0.8, color='lightgreen')
        ax.set_xlabel('Method')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Iterations to converge
        ax = axes[1, 1]
        iterations = [self.results[m]['convergence_info']['n_iterations'] for m in methods]
        bars = ax.bar(methods, iterations, alpha=0.8, color='purple')
        ax.set_xlabel('Method')
        ax.set_ylabel('Iterations')
        ax.set_title('Iterations to Converge')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, iter_count in zip(bars, iterations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{iter_count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(REPORTS_DIR / 'gradient_descent_performance.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        self.console.print("[green]✓ Performance comparison plot saved[/green]")
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the comparison."""
        report = []
        report.append("# Gradient Descent Methods Comparison Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset information
        report.append("## Dataset Information\n")
        report.append(f"- Training samples: {len(self.X_train):,}\n")
        report.append(f"- Test samples: {len(self.X_test):,}\n")
        report.append(f"- Number of features: {self.X_train.shape[1]}\n")
        
        # Results summary
        report.append("\n## Results Summary\n")
        
        for method, result in self.results.items():
            report.append(f"\n### {method.upper()} Gradient Descent\n")
            
            # Hyperparameters
            report.append("**Hyperparameters:**\n")
            for param, value in result['hyperparameters'].items():
                if value is not None:
                    report.append(f"- {param}: {value}\n")
            
            # Metrics
            report.append("\n**Performance Metrics:**\n")
            metrics = result['metrics']
            report.append(f"- Train RMSE: {metrics['train_rmse']:.4f}\n")
            report.append(f"- Test RMSE: {metrics['test_rmse']:.4f}\n")
            report.append(f"- Train MAE: {metrics['train_mae']:.4f}\n")
            report.append(f"- Test MAE: {metrics['test_mae']:.4f}\n")
            report.append(f"- Train R²: {metrics['train_r2']:.4f}\n")
            report.append(f"- Test R²: {metrics['test_r2']:.4f}\n")
            
            # Convergence info
            report.append("\n**Convergence Information:**\n")
            conv_info = result['convergence_info']
            report.append(f"- Training time: {conv_info['training_time']:.2f} seconds\n")
            report.append(f"- Iterations: {conv_info['n_iterations']}\n")
            report.append(f"- Converged: {'Yes' if conv_info['converged'] else 'No'}\n")
            report.append(f"- Final loss: {conv_info['final_loss']:.6f}\n")
        
        # Best method
        best_method = min(self.results.items(), key=lambda x: x[1]['metrics']['test_rmse'])
        report.append(f"\n## Best Method: {best_method[0].upper()}\n")
        report.append(f"- Test RMSE: {best_method[1]['metrics']['test_rmse']:.4f}\n")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        report.append("Based on the results:\n")
        report.append("- **Batch GD**: Best for small datasets where memory is not a constraint\n")
        report.append("- **SGD**: Best for online learning and very large datasets\n")
        report.append("- **Mini-Batch GD**: Best balance between convergence stability and speed\n")
        
        report_text = ''.join(report)
        
        # Save report
        report_path = REPORTS_DIR / 'gradient_descent_comparison_report.md'
        report_path.write_text(report_text)
        self.console.print(f"[green]✓ Report saved to {report_path}[/green]")
        
        return report_text
    
    def save_models(self):
        """Save trained models."""
        models_dir = PROCESSED_DATA_DIR / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for method, model in self.models.items():
            model_path = models_dir / f'gradient_descent_{method}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': self.scaler,
                    'results': self.results[method]
                }, f)
        
        self.console.print(f"[green]✓ Models saved to {models_dir}[/green]")


def run_regression_pipeline(sample_size: Optional[int] = None) -> RatingPredictionPipeline:
    """
    Run the complete regression pipeline with gradient descent comparison.
    
    Args:
        sample_size: Optional sample size for faster testing
        
    Returns:
        Fitted pipeline object
    """
    console.print(Panel.fit(
        "[bold cyan]Rating Prediction with Gradient Descent Variants[/bold cyan]\n"
        "Comparing Batch, Stochastic, and Mini-Batch Gradient Descent",
        border_style="cyan"
    ))
    
    # Initialize pipeline
    pipeline = RatingPredictionPipeline()
    
    # Load data
    try:
        X, y = pipeline.load_ml_dataset()
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        console.print("[yellow]Run 'python analyze.py preprocess' first[/yellow]")
        return None
    
    # Sample data if requested
    if sample_size and sample_size < len(X):
        console.print(f"[yellow]Sampling {sample_size:,} examples for faster testing[/yellow]")
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    # Prepare data
    pipeline.prepare_data(X, y)
    
    # Compare methods
    pipeline.compare_gradient_descent_methods()
    
    # Display results
    pipeline.display_comparison_results()
    
    # Generate visualizations
    REPORTS_DIR.mkdir(exist_ok=True)
    pipeline.plot_convergence_curves()
    pipeline.plot_performance_comparison()
    
    # Generate report
    pipeline.generate_report()
    
    # Save models
    pipeline.save_models()
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    console.print("[yellow]Running regression pipeline with gradient descent comparison...[/yellow]")
    pipeline = run_regression_pipeline(sample_size=50000)  # Use sample for testing