import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.sparse import csr_matrix
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from ..config import *

class DataTransformer:
    """Advanced data transformation and feature engineering utilities."""
    
    def __init__(self):
        self.console = Console()
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.feature_selectors = {}
        self.transformation_log = []
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values with different strategies."""
        self.console.print("[cyan]Handling missing values...[/cyan]")
        
        df_clean = df.copy()
        missing_info = []
        
        for column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            missing_pct = (missing_count / len(df_clean)) * 100
            
            if missing_count > 0:
                missing_info.append({
                    'column': column,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct
                })
                
                if strategy == 'auto':
                    if missing_pct > 50:
                        # Drop columns with more than 50% missing
                        df_clean = df_clean.drop(columns=[column])
                        self.console.print(f"[yellow]Dropped {column} (>{missing_pct:.1f}% missing)[/yellow]")
                    elif df_clean[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                        # Fill numerical columns with median
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                        self.console.print(f"[green]Filled {column} with median[/green]")
                    else:
                        # Fill categorical columns with mode
                        mode_value = df_clean[column].mode()
                        if len(mode_value) > 0:
                            df_clean[column] = df_clean[column].fillna(mode_value[0])
                            self.console.print(f"[green]Filled {column} with mode[/green]")
                        else:
                            df_clean[column] = df_clean[column].fillna('Unknown')
                            self.console.print(f"[green]Filled {column} with 'Unknown'[/green]")
        
        # Display missing value summary
        if missing_info:
            table = Table(title="Missing Values Summary")
            table.add_column("Column", style="cyan")
            table.add_column("Missing Count", justify="right")
            table.add_column("Missing %", justify="right")
            
            for info in missing_info:
                table.add_row(
                    info['column'],
                    str(info['missing_count']),
                    f"{info['missing_percentage']:.2f}%"
                )
            self.console.print(table)
        else:
            self.console.print("[green]✓ No missing values found[/green]")
            
        self.transformation_log.append(f"Missing values handled using {strategy} strategy")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate records."""
        self.console.print("[cyan]Checking for duplicates...[/cyan]")
        
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        if duplicates_removed > 0:
            self.console.print(f"[yellow]Removed {duplicates_removed} duplicate records[/yellow]")
        else:
            self.console.print("[green]✓ No duplicates found[/green]")
            
        self.transformation_log.append(f"Removed {duplicates_removed} duplicate records")
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers."""
        self.console.print(f"[cyan]Detecting outliers using {method} method...[/cyan]")
        
        df_clean = df.copy()
        outlier_info = []
        
        for column in columns:
            if column in df_clean.columns and df_clean[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                if method == 'iqr':
                    Q1 = df_clean[column].quantile(0.25)
                    Q3 = df_clean[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                    outlier_count = outliers_mask.sum()
                    
                    if outlier_count > 0:
                        # Cap outliers instead of removing them
                        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
                        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
                        
                        outlier_info.append({
                            'column': column,
                            'outlier_count': outlier_count,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        })
        
        # Display outlier summary
        if outlier_info:
            table = Table(title="Outliers Detected and Capped")
            table.add_column("Column", style="cyan")
            table.add_column("Outliers", justify="right")
            table.add_column("Lower Bound", justify="right")
            table.add_column("Upper Bound", justify="right")
            
            for info in outlier_info:
                table.add_row(
                    info['column'],
                    str(info['outlier_count']),
                    f"{info['lower_bound']:.2f}",
                    f"{info['upper_bound']:.2f}"
                )
            self.console.print(table)
        else:
            self.console.print("[green]✓ No significant outliers detected[/green]")
            
        return df_clean
    
    def create_user_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive user-based features."""
        self.console.print("[cyan]Creating user features...[/cyan]")
        
        user_features = ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique',
            'timestamp': ['min', 'max']
        })
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Additional derived features
        user_features['rating_std'] = user_features['rating_std'].fillna(0)
        user_features['activity_span_days'] = (
            user_features['timestamp_max'] - user_features['timestamp_min']
        ).dt.total_seconds() / (24 * 3600)
        
        user_features['rating_frequency'] = (
            user_features['rating_count'] / (user_features['activity_span_days'] + 1)
        )
        
        # User rating patterns
        user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
        user_features['is_active_user'] = (user_features['rating_count'] > user_features['rating_count'].quantile(0.75)).astype(int)
        user_features['is_picky_user'] = (user_features['rating_std'] > user_features['rating_std'].quantile(0.75)).astype(int)
        
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features[/green]")
        return user_features
    
    def create_movie_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive movie-based features."""
        self.console.print("[cyan]Creating movie features...[/cyan]")
        
        # Basic movie statistics
        movie_features = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique',
            'timestamp': ['min', 'max']
        })
        
        movie_features.columns = ['_'.join(col).strip() for col in movie_features.columns.values]
        
        # Merge with movie metadata
        movie_features = movie_features.join(movies_df.set_index('movieId'), how='left')
        
        # Genre features
        if 'genre_list' in movie_features.columns:
            # Count genres per movie
            movie_features['genre_count'] = movie_features['genre_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Create binary features for top genres
            all_genres = []
            for genre_list in movie_features['genre_list'].dropna():
                if isinstance(genre_list, list):
                    all_genres.extend(genre_list)
            
            top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
            
            for genre in top_genres:
                if genre != '(no genres listed)':
                    movie_features[f'is_{genre.lower().replace(" ", "_").replace("-", "_")}'] = (
                        movie_features['genre_list'].apply(
                            lambda x: 1 if isinstance(x, list) and genre in x else 0
                        )
                    )
        
        # Movie age and popularity features
        current_year = pd.Timestamp.now().year
        movie_features['movie_age'] = current_year - movie_features['year'].fillna(current_year)
        movie_features['popularity_score'] = (
            movie_features['rating_count'] * movie_features['rating_mean']
        ).fillna(0)
        
        # Rating patterns
        movie_features['rating_std'] = movie_features['rating_std'].fillna(0)
        movie_features['is_popular'] = (
            movie_features['rating_count'] > movie_features['rating_count'].quantile(0.8)
        ).astype(int)
        movie_features['is_highly_rated'] = (
            movie_features['rating_mean'] > movie_features['rating_mean'].quantile(0.8)
        ).astype(int)
        
        self.console.print(f"[green]✓ Created {len(movie_features.columns)} movie features[/green]")
        return movie_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str], 
                                  method: str = 'label') -> pd.DataFrame:
        """Encode categorical features using various methods."""
        self.console.print(f"[cyan]Encoding categorical features using {method} encoding...[/cyan]")
        
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                if method == 'label':
                    le = LabelEncoder()
                    df_encoded[f'{column}_encoded'] = le.fit_transform(df_encoded[column].astype(str))
                    self.encoders[f'{column}_label'] = le
                    
                elif method == 'onehot':
                    # Use get_dummies for simplicity
                    dummy_df = pd.get_dummies(df_encoded[column], prefix=column)
                    df_encoded = pd.concat([df_encoded, dummy_df], axis=1)
                    
                self.console.print(f"[green]✓ Encoded {column}[/green]")
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, numerical_columns: List[str], 
                          method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features."""
        self.console.print(f"[cyan]Normalizing features using {method} scaling...[/cyan]")
        
        df_normalized = df.copy()
        
        for column in numerical_columns:
            if column in df_normalized.columns:
                if method == 'standard':
                    scaler = StandardScaler()
                    df_normalized[f'{column}_scaled'] = scaler.fit_transform(
                        df_normalized[[column]]
                    ).flatten()
                    self.scalers[f'{column}_standard'] = scaler
                    
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                    df_normalized[f'{column}_scaled'] = scaler.fit_transform(
                        df_normalized[[column]]
                    ).flatten()
                    self.scalers[f'{column}_minmax'] = scaler
                    
                elif method == 'robust':
                    # Manual robust scaling (median and IQR)
                    median = df_normalized[column].median()
                    q75 = df_normalized[column].quantile(0.75)
                    q25 = df_normalized[column].quantile(0.25)
                    iqr = q75 - q25
                    
                    if iqr > 0:
                        df_normalized[f'{column}_robust'] = (df_normalized[column] - median) / iqr
                    else:
                        df_normalized[f'{column}_robust'] = 0
                
                self.console.print(f"[green]✓ Normalized {column}[/green]")
        
        return df_normalized
    
    def apply_pca(self, df: pd.DataFrame, feature_columns: List[str], 
                  n_components: Optional[int] = None, variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply PCA for dimensionality reduction."""
        self.console.print("[cyan]Applying PCA for dimensionality reduction...[/cyan]")
        
        # Select numerical features for PCA
        feature_data = df[feature_columns].select_dtypes(include=[np.number]).fillna(0)
        
        if n_components is None:
            # Determine optimal number of components
            pca_temp = PCA()
            pca_temp.fit(feature_data)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(feature_data)
        
        # Create PCA dataframe
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
        
        # Store PCA model
        self.pca_models['main_pca'] = pca
        
        # Create explained variance summary
        explained_variance = pd.DataFrame({
            'Component': pca_columns,
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        self.console.print(f"[green]✓ Reduced {len(feature_columns)} features to {n_components} components[/green]")
        self.console.print(f"[green]✓ Explained variance: {pca.explained_variance_ratio_.sum():.3f}[/green]")
        
        return pca_df, explained_variance
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified feature pairs."""
        self.console.print("[cyan]Creating interaction features...[/cyan]")
        
        df_interactions = df.copy()
        created_features = []
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df_interactions.columns and feature2 in df_interactions.columns:
                # Multiplication interaction
                interaction_name = f'{feature1}_x_{feature2}'
                df_interactions[interaction_name] = df_interactions[feature1] * df_interactions[feature2]
                created_features.append(interaction_name)
                
                # Ratio interaction (if feature2 is not zero)
                if (df_interactions[feature2] != 0).all():
                    ratio_name = f'{feature1}_div_{feature2}'
                    df_interactions[ratio_name] = df_interactions[feature1] / df_interactions[feature2]
                    created_features.append(ratio_name)
        
        self.console.print(f"[green]✓ Created {len(created_features)} interaction features[/green]")
        return df_interactions
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 'auto', method: str = 'univariate') -> pd.DataFrame:
        """Select the most important features."""
        self.console.print(f"[cyan]Selecting features using {method} method...[/cyan]")
        
        if k == 'auto':
            k = min(50, X.shape[1] // 2)  # Select up to 50 features or half of available features
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.feature_selectors['univariate'] = selector
            
        self.console.print(f"[green]✓ Selected {len(selected_features)} most important features[/green]")
        return X_selected_df
    
    def create_user_item_matrix(self, ratings_df: pd.DataFrame, sparse: bool = True) -> Tuple[np.ndarray, Dict, Dict]:
        """Create user-item matrix for collaborative filtering."""
        self.console.print("[cyan]Creating user-item matrix...[/cyan]")
        
        # Create mappings
        users = sorted(ratings_df['userId'].unique())
        movies = sorted(ratings_df['movieId'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(movies)}
        
        # Create matrix
        if sparse:
            from scipy.sparse import lil_matrix
            matrix = lil_matrix((len(users), len(movies)))
            
            for _, row in ratings_df.iterrows():
                user_idx = user_to_idx[row['userId']]
                movie_idx = movie_to_idx[row['movieId']]
                matrix[user_idx, movie_idx] = row['rating']
            
            matrix = matrix.tocsr()  # Convert to CSR for efficient operations
            
            # Fix: Use matrix.nnz for sparse matrices
            total_elements = matrix.shape[0] * matrix.shape[1]
            sparsity = 1 - (matrix.nnz / total_elements)
            
        else:
            matrix = np.zeros((len(users), len(movies)))
            
            for _, row in ratings_df.iterrows():
                user_idx = user_to_idx[row['userId']]
                movie_idx = movie_to_idx[row['movieId']]
                matrix[user_idx, movie_idx] = row['rating']
            
            # For dense matrices, use np.count_nonzero
            total_elements = matrix.shape[0] * matrix.shape[1]
            sparsity = 1 - (np.count_nonzero(matrix) / total_elements)
        
        self.console.print(f"[green]✓ Created {matrix.shape[0]}x{matrix.shape[1]} user-item matrix[/green]")
        self.console.print(f"[green]✓ Sparsity: {sparsity:.4f}[/green]")
        
        return matrix, user_to_idx, movie_to_idx
    def get_transformation_summary(self) -> Dict:
        """Get summary of all transformations applied."""
        return {
            'transformations_applied': self.transformation_log,
            'scalers_created': list(self.scalers.keys()),
            'encoders_created': list(self.encoders.keys()),
            'pca_models_created': list(self.pca_models.keys()),
            'feature_selectors_created': list(self.feature_selectors.keys())
        }
    
    def save_transformers(self, filepath: str):
        """Save all transformation objects for later use."""
        import pickle
        
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca_models': self.pca_models,
            'feature_selectors': self.feature_selectors,
            'transformation_log': self.transformation_log
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(transformers, f)
        
        self.console.print(f"[green]✓ Transformers saved to {filepath}[/green]")
    
    def load_transformers(self, filepath: str):
        """Load previously saved transformation objects."""
        import pickle
        
        with open(filepath, 'rb') as f:
            transformers = pickle.load(f)
        
        self.scalers = transformers.get('scalers', {})
        self.encoders = transformers.get('encoders', {})
        self.pca_models = transformers.get('pca_models', {})
        self.feature_selectors = transformers.get('feature_selectors', {})
        self.transformation_log = transformers.get('transformation_log', [])
        
        self.console.print(f"[green]✓ Transformers loaded from {filepath}[/green]")