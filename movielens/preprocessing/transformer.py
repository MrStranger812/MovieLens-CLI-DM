import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.sparse import csr_matrix
from scipy import sparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from ..config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

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
    def create_temporal_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create advanced temporal features for better predictions."""
        self.console.print("[cyan]Creating temporal features...[/cyan]")
        
        temp_df = ratings_df.copy()
        
        # Check if required columns exist
        if 'timestamp' not in temp_df.columns:
            self.console.print("[red]Error: timestamp column not found in ratings data[/red]")
            return temp_df, pd.DataFrame(), pd.DataFrame()
        
        # 1. User rating evolution features
        try:
            user_temporal = temp_df.groupby('userId').apply(
                lambda x: pd.Series({
                    'first_rating_date': x['timestamp'].min(),
                    'last_rating_date': x['timestamp'].max(),
                    'active_days': (x['timestamp'].max() - x['timestamp'].min()).days,
                    'rating_velocity': len(x) / max((x['timestamp'].max() - x['timestamp'].min()).days, 1),
                    'rating_acceleration': self._calculate_rating_acceleration(x),
                    'rating_trend': self._calculate_rating_trend(x),
                    'weekend_preference': (x['day_of_week'].isin(['Saturday', 'Sunday']).sum() / len(x)) if 'day_of_week' in x.columns else 0,
                    'night_owl_score': (x['hour'].between(22, 23) | x['hour'].between(0, 5)).sum() / len(x) if 'hour' in x.columns else 0
                })
            )
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create user temporal features: {e}[/yellow]")
            user_temporal = pd.DataFrame()
        
        # 2. Movie popularity dynamics - handle year column properly
        try:
            # Get year from movies if not in ratings
            if 'year' not in temp_df.columns:
                movies_years = ratings_df[['movieId']].drop_duplicates().merge(
                    movies_df[['movieId', 'year']], on='movieId', how='left'
                )
                temp_df = temp_df.merge(movies_years, on='movieId', how='left')
            
            movie_temporal = temp_df.groupby('movieId').apply(
                lambda x: pd.Series({
                    'release_to_first_rating': (
                        (x['timestamp'].min() - pd.Timestamp(f"{int(x['year'].iloc[0])}-01-01")).days 
                        if pd.notna(x['year'].iloc[0]) and x['year'].iloc[0] > 1900 
                        else None
                    ),
                    'rating_momentum': self._calculate_momentum(x),
                    'rating_volatility': x.set_index('timestamp')['rating'].rolling('30D').std().mean() if len(x) > 1 else 0,
                    'peak_rating_period': x.groupby(pd.Grouper(key='timestamp', freq='M'))['rating'].mean().idxmax() if len(x) > 0 else None,
                    'rating_decay_rate': self._calculate_decay_rate(x)
                })
            )
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create movie temporal features: {e}[/yellow]")
            movie_temporal = pd.DataFrame()
        
        # Rest of the method remains the same...
        # 3. Seasonal patterns
        temp_df['season'] = temp_df['timestamp'].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # 4. Holiday effects (US holidays)
        holidays = [
            ('2010-12-25', '2014-12-25'),  # Christmas
            ('2010-01-01', '2014-01-01'),  # New Year
            ('2010-07-04', '2014-07-04'),  # Independence Day
        ]
        
        temp_df['is_holiday_week'] = 0
        for start, end in holidays:
            holiday_range = pd.date_range(start=start, end=end, freq='YS')
            for holiday in holiday_range:
                mask = (temp_df['timestamp'] >= holiday - pd.Timedelta(days=3)) & \
                    (temp_df['timestamp'] <= holiday + pd.Timedelta(days=3))
                temp_df.loc[mask, 'is_holiday_week'] = 1
        
        self.console.print("[green]✓ Created temporal features[/green]")
        return temp_df, user_temporal, movie_temporal

    def _calculate_rating_trend(self, user_data: pd.DataFrame) -> float:
        """Calculate if user's ratings are trending up or down over time."""
        if len(user_data) < 10:
            return 0.0
        
        sorted_data = user_data.sort_values('timestamp')
        x = np.arange(len(sorted_data))
        y = sorted_data['rating'].values
        
        # Simple linear regression for trend
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _calculate_rating_acceleration(self, user_data: pd.DataFrame) -> float:
        """Calculate how fast user's rating frequency is changing."""
        if len(user_data) < 20:
            return 0.0
        
        sorted_data = user_data.sort_values('timestamp')
        # Group by month and count ratings
        monthly_counts = sorted_data.groupby(pd.Grouper(key='timestamp', freq='M')).size()
        
        if len(monthly_counts) < 3:
            return 0.0
        
        # Calculate acceleration as change in rating velocity
        velocities = monthly_counts.diff()
        acceleration = velocities.diff().mean()
        return acceleration if pd.notna(acceleration) else 0.0

    def _calculate_momentum(self, movie_data: pd.DataFrame) -> float:
        """Calculate movie's rating momentum (recent ratings vs older ratings)."""
        if len(movie_data) < 10:
            return 0.0
        
        sorted_data = movie_data.sort_values('timestamp')
        midpoint = len(sorted_data) // 2
        
        recent_avg = sorted_data.iloc[midpoint:]['rating'].mean()
        older_avg = sorted_data.iloc[:midpoint]['rating'].mean()
        
        return recent_avg - older_avg

    def _calculate_decay_rate(self, movie_data: pd.DataFrame) -> float:
        """Calculate how fast movie ratings decay over time."""
        if len(movie_data) < 20:
            return 0.0
        
        sorted_data = movie_data.sort_values('timestamp')
        
        # Group by quarter and calculate average rating
        quarterly_avg = sorted_data.groupby(pd.Grouper(key='timestamp', freq='Q'))['rating'].mean()
        
        if len(quarterly_avg) < 4:
            return 0.0
        
        # Fit exponential decay: rating = a * exp(-b * time)
        x = np.arange(len(quarterly_avg))
        y = quarterly_avg.values
        
        # Log transform for linear fitting
        log_y = np.log(y + 1e-10)  # Add small constant to avoid log(0)
        slope, _ = np.polyfit(x, log_y, 1)
        
        return -slope  # Positive value means decay
    
    def create_tag_features(self, tags_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Create features from movie tags using TF-IDF and other text processing techniques."""
        self.console.print("[cyan]Creating tag-based features...[/cyan]")
        
        # 1. Aggregate tags by movie
        movie_tags = tags_df.groupby('movieId')['tag_clean'].apply(lambda x: ' '.join(x)).reset_index()
        movie_tags.columns = ['movieId', 'combined_tags']
        
        # 2. Basic tag statistics per movie
        tag_stats = tags_df.groupby('movieId').agg({
            'tag_clean': ['count', 'nunique'],
            'userId': 'nunique'
        })
        tag_stats.columns = ['total_tags', 'unique_tags', 'users_who_tagged']
        tag_stats['tag_diversity'] = tag_stats['unique_tags'] / tag_stats['total_tags']
        
        # 3. TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),  # Include bigrams
            min_df=5,  # Tag must appear in at least 5 movies
            max_df=0.7  # Ignore tags that appear in >70% of movies
        )
        
        # Fill missing movies with empty strings
        all_movies = pd.DataFrame({'movieId': movies_df['movieId']})
        movie_tags_complete = all_movies.merge(movie_tags, on='movieId', how='left')
        movie_tags_complete['combined_tags'] = movie_tags_complete['combined_tags'].fillna('')
        
        tfidf_matrix = tfidf.fit_transform(movie_tags_complete['combined_tags'])
        
        # Convert to DataFrame
        tfidf_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()],
            index=movie_tags_complete['movieId']
        )
        
        # 4. Tag sentiment features (using simple keyword matching)
        positive_words = ['excellent', 'amazing', 'great', 'love', 'wonderful', 'best', 'fantastic', 'masterpiece']
        negative_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'boring', 'waste', 'disappointing']
        
        def calculate_sentiment(tags):
            if pd.isna(tags) or tags == '':
                return {'positive_score': 0, 'negative_score': 0, 'sentiment_ratio': 0}
            
            tags_lower = tags.lower()
            positive_count = sum(word in tags_lower for word in positive_words)
            negative_count = sum(word in tags_lower for word in negative_words)
            total = positive_count + negative_count
            
            return {
                'positive_score': positive_count,
                'negative_score': negative_count,
                'sentiment_ratio': (positive_count - negative_count) / (total + 1)
            }
        
        sentiment_features = movie_tags_complete['combined_tags'].apply(
            lambda x: pd.Series(calculate_sentiment(x))
        )
        sentiment_features.index = movie_tags_complete['movieId']
        
        # 5. Tag topic modeling features (using simple clustering of TF-IDF)
        from sklearn.decomposition import LatentDirichletAllocation
        
        # Create topic model
        n_topics = 10
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_matrix = lda.fit_transform(tfidf_matrix)
        
        # Convert to DataFrame
        topic_features = pd.DataFrame(
            topic_matrix,
            columns=[f'topic_{i}' for i in range(n_topics)],
            index=movie_tags_complete['movieId']
        )
        
        # 6. Combine all tag features
        tag_features = pd.concat([
            tag_stats,
            tfidf_features.iloc[:, :50],  # Keep top 50 TF-IDF features
            sentiment_features,
            topic_features
        ], axis=1)
        
        # Store the vectorizers for later use
        tag_transformers = {
            'tfidf': tfidf,
            'lda': lda
        }
        
        self.console.print(f"[green]✓ Created {tag_features.shape[1]} tag-based features[/green]")
        
        return tag_features, tag_transformers

    def create_user_tag_preferences(self, ratings_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Create user preference profiles based on tags of movies they rated highly."""
        self.console.print("[cyan]Creating user tag preference features...[/cyan]")
        
        # Get movies that users rated highly (>= 4.0)
        high_ratings = ratings_df[ratings_df['rating'] >= 4.0][['userId', 'movieId']]
        
        # Create a movie-to-tags mapping
        movie_tags_dict = tags_df.groupby('movieId')['tag_clean'].apply(lambda x: ' '.join(x)).to_dict()
        
        # Get unique users
        user_ids = high_ratings['userId'].unique()
        batch_size = 10000  # Adjust based on available memory
        n_batches = (len(user_ids) - 1) // batch_size + 1
        
        # Initialize TF-IDF vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        user_tfidf = TfidfVectorizer(max_features=50, min_df=2)
        
        # Fit the vectorizer on a sample of movie tags to define vocabulary
        sample_size = min(10000, len(high_ratings))
        sample_tags = [movie_tags_dict.get(mid, '') for mid in high_ratings['movieId'].sample(sample_size)]
        user_tfidf.fit(sample_tags)
        
        # Process users in batches
        user_tag_features_list = []
        
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            batch_ratings = high_ratings[high_ratings['userId'].isin(batch_users)]
            
            # Aggregate tags for each user's highly rated movies
            user_tags = batch_ratings.groupby('userId')['movieId'].apply(
                lambda x: ' '.join([movie_tags_dict.get(mid, '') for mid in x])
            )
            
            # Transform to TF-IDF features
            batch_tfidf = user_tfidf.transform(user_tags)
            
            # Create a sparse DataFrame for this batch
            batch_features = pd.DataFrame.sparse.from_spmatrix(
                batch_tfidf,
                columns=[f'user_pref_{word}' for word in user_tfidf.get_feature_names_out()],
                index=user_tags.index
            )
            
            user_tag_features_list.append(batch_features)
            
            # Free memory after each batch
            del batch_ratings, user_tags, batch_tfidf
            import gc
            gc.collect()
        
        # Combine all batch features
        user_tag_features = pd.concat(user_tag_features_list)
        
        self.console.print(f"[green]✓ Created {user_tag_features.shape[1]} user preference features[/green]")
        
        return user_tag_features
    
    def create_enhanced_genre_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive genre-based features."""
        self.console.print("[cyan]Creating enhanced genre features...[/cyan]")
        
        from sklearn.preprocessing import MultiLabelBinarizer
        
        # 1. Binary encoding for all genres
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies_df['genre_list'])
        
        genre_binary = pd.DataFrame(
            genre_matrix,
            columns=[f'genre_{genre}'.lower().replace(' ', '_').replace('-', '_') 
                    for genre in mlb.classes_],
            index=movies_df['movieId']
        )
        
        # 2. Genre combination features (for common genre pairs)
        # Find most common genre combinations
        genre_combos = movies_df['genre_list'].apply(
            lambda x: [f"{g1}_{g2}" for i, g1 in enumerate(x) for g2 in x[i+1:]] 
            if isinstance(x, list) else []
        )
        
        all_combos = [combo for combos in genre_combos if combos for combo in combos]
        combo_counts = pd.Series(all_combos).value_counts().head(20)
        
        # Create binary features for top genre combinations
        for combo in combo_counts.index:
            g1, g2 = combo.split('_')
            genre_binary[f'combo_{combo}'.lower()] = (
                movies_df['genre_list'].apply(
                    lambda x: 1 if isinstance(x, list) and g1 in x and g2 in x else 0
                ).values
            )
        
        # 3. Genre statistics per movie
        genre_stats = pd.DataFrame(index=movies_df['movieId'])
        genre_stats['genre_count'] = movies_df['genre_list'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        genre_stats['is_single_genre'] = (genre_stats['genre_count'] == 1).astype(int)
        genre_stats['is_multi_genre'] = (genre_stats['genre_count'] > 1).astype(int)
        genre_stats['genre_diversity'] = movies_df['genre_list'].apply(
            lambda x: len(set(x)) / len(x) if isinstance(x, list) and len(x) > 0 else 0
        )
        
        # 4. Genre popularity features (based on ratings)
        movie_ratings = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        })
        movie_ratings.columns = ['avg_rating', 'num_ratings', 'rating_std']
        
        # Calculate genre performance metrics
        genre_performance = {}
        for genre in mlb.classes_:
            if genre != '(no genres listed)':
                genre_movies = movies_df[movies_df['genre_list'].apply(
                    lambda x: genre in x if isinstance(x, list) else False
                )]['movieId']
                
                genre_ratings = movie_ratings[movie_ratings.index.isin(genre_movies)]
                
                genre_performance[genre] = {
                    'avg_genre_rating': genre_ratings['avg_rating'].mean(),
                    'genre_popularity': genre_ratings['num_ratings'].sum(),
                    'genre_rating_consistency': genre_ratings['rating_std'].mean()
                }
        
        # Add genre performance scores to each movie
        for col_type in ['avg_genre_rating', 'genre_popularity', 'genre_rating_consistency']:
            genre_stats[f'{col_type}_score'] = movies_df['genre_list'].apply(
                lambda genres: np.mean([
                    genre_performance.get(g, {}).get(col_type, 0) 
                    for g in genres if isinstance(genres, list) and g in genre_performance
                ]) if isinstance(genres, list) and len(genres) > 0 else 0
            )
        
        # 5. Genre rarity features
        genre_counts = pd.Series([g for genres in movies_df['genre_list'] 
                                for g in genres if isinstance(genres, list)]).value_counts()
        
        genre_stats['genre_rarity_score'] = movies_df['genre_list'].apply(
            lambda genres: np.mean([
                1 / (genre_counts.get(g, 1) + 1) 
                for g in genres if isinstance(genres, list)
            ]) if isinstance(genres, list) and len(genres) > 0 else 0
        )
        
        # Combine all genre features
        genre_features = pd.concat([genre_binary, genre_stats], axis=1)
        
        self.console.print(f"[green]✓ Created {genre_features.shape[1]} genre features[/green]")
        
        return genre_features
    
    def create_cold_start_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                               user_features: pd.DataFrame, movie_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features to handle cold start problem for users and movies with few ratings."""
        self.console.print("[cyan]Creating cold start handling features...[/cyan]")
        
        # 1. User cold start features
        user_cold_features = pd.DataFrame(index=user_features.index)
        
        # Identify cold start users
        user_rating_counts = ratings_df['userId'].value_counts()
        cold_threshold = user_rating_counts.quantile(0.1)  # Bottom 10% are cold start
        
        user_cold_features['is_cold_start'] = (user_features['rating_count'] < cold_threshold).astype(int)
        user_cold_features['rating_confidence'] = 1 - np.exp(-user_features['rating_count'] / 10)  # Confidence score
        
        # For cold start users, use demographic/time-based features
        cold_users = user_cold_features[user_cold_features['is_cold_start'] == 1].index
        
        for user_id in cold_users:
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            # Time of day preferences (more reliable with few ratings)
            if len(user_ratings) > 0:
                user_cold_features.loc[user_id, 'preferred_hour'] = user_ratings['hour'].mode()[0] if 'hour' in user_ratings.columns else 12
                user_cold_features.loc[user_id, 'preferred_day'] = user_ratings['day_of_week'].mode()[0] if 'day_of_week' in user_ratings.columns else 'Friday'
        
        # Similar user clustering for cold start users
        # Use time-based features which are available even with few ratings
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        time_features = ['rating_count', 'activity_span_days', 'rating_frequency']
        available_time_features = [f for f in time_features if f in user_features.columns]
        
        if available_time_features:
            scaler = StandardScaler()
            user_time_scaled = scaler.fit_transform(user_features[available_time_features].fillna(0))
            
            # Cluster users into behavioral groups
            n_clusters = min(20, len(user_features) // 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            user_cold_features['user_cluster'] = kmeans.fit_predict(user_time_scaled)
            
            # Get cluster statistics for cold start predictions
            cluster_stats = ratings_df.merge(
                user_cold_features[['user_cluster']], 
                left_on='userId', 
                right_index=True
            ).groupby('user_cluster')['rating'].agg(['mean', 'std'])
            
            user_cold_features['cluster_avg_rating'] = user_cold_features['user_cluster'].map(cluster_stats['mean'])
            user_cold_features['cluster_rating_std'] = user_cold_features['user_cluster'].map(cluster_stats['std'])
        
        # 2. Movie cold start features
        movie_cold_features = pd.DataFrame(index=movie_features.index)
        
        # Identify cold start movies
        movie_rating_counts = ratings_df['movieId'].value_counts()
        movie_cold_threshold = movie_rating_counts.quantile(0.1)
        
        movie_cold_features['is_cold_start'] = (movie_features['rating_count'] < movie_cold_threshold).astype(int)
        movie_cold_features['rating_confidence'] = 1 - np.exp(-movie_features['rating_count'] / 10)
        
        # For cold start movies, rely more on content features
        cold_movies = movie_cold_features[movie_cold_features['is_cold_start'] == 1].index
        
        
        # Use genre-based predictions for cold movies
        if 'genres' in movies_df.columns:
            genre_avg_ratings = {}
            for genre in movies_df['genres'].str.split('|').explode().unique():
                if genre and genre != '(no genres listed)':
                    genre_movies = movies_df[movies_df['genres'].str.contains(genre, na=False)]['movieId']
                    genre_ratings = ratings_df[ratings_df['movieId'].isin(genre_movies)]
                    if len(genre_ratings) > 0:
                        genre_avg_ratings[genre] = genre_ratings['rating'].mean()
            
            # Calculate expected rating based on genres - match index properly
            genre_expected_ratings = movies_df.set_index('movieId')['genres'].apply(
                lambda x: np.mean([genre_avg_ratings.get(g, 3.5) for g in x.split('|') if g in genre_avg_ratings]) 
                if pd.notna(x) else 3.5
            )
    
            # Only assign for movies that exist in movie_cold_features
            movie_cold_features['genre_expected_rating'] = genre_expected_ratings.reindex(movie_cold_features.index, fill_value=3.5)
                
        # Similar movie clustering for cold start movies
        content_features = [col for col in movie_features.columns 
                        if col.startswith('genre_') or col.startswith('is_')]
        
        if content_features:
            movie_content_scaled = StandardScaler().fit_transform(
                movie_features[content_features].fillna(0)
            )
            
            # Cluster movies by content
            movie_kmeans = KMeans(n_clusters=min(50, len(movie_features) // 100), random_state=42)
            movie_cold_features['movie_cluster'] = movie_kmeans.fit_predict(movie_content_scaled)
            
            # Get cluster statistics
            movie_cluster_stats = ratings_df.merge(
                movie_cold_features[['movie_cluster']], 
                left_on='movieId', 
                right_index=True
            ).groupby('movie_cluster')['rating'].agg(['mean', 'std', 'count'])
            
            movie_cold_features['cluster_avg_rating'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['mean'])
            movie_cold_features['cluster_rating_std'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['std'])
            movie_cold_features['cluster_popularity'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['count'])
        
        # 3. Fallback features for extreme cold start
        user_cold_features['global_avg_rating'] = ratings_df['rating'].mean()
        movie_cold_features['global_avg_rating'] = ratings_df['rating'].mean()
        
        self.console.print(f"[green]✓ Created {user_cold_features.shape[1]} user cold start features[/green]")
        self.console.print(f"[green]✓ Created {movie_cold_features.shape[1]} movie cold start features[/green]")
        
        return user_cold_features, movie_cold_features
    
    def validate_features(self, feature_df: pd.DataFrame, feature_type: str = "general") -> Dict:
        """Validate feature quality and identify potential issues."""
        self.console.print(f"[cyan]Validating {feature_type} features...[/cyan]")
        
        validation_results = {
            'total_features': len(feature_df.columns),
            'issues': [],
            'warnings': [],
            'feature_stats': {}
        }
        
        # 1. Check for constant features (no variance)
        constant_features = []
        for col in feature_df.select_dtypes(include=[np.number]).columns:
            if feature_df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            validation_results['issues'].append(f"Found {len(constant_features)} constant features: {constant_features[:5]}")
        
        # 2. Check for highly correlated features
        numeric_features = feature_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            highly_correlated = []
            for column in upper_triangle.columns:
                correlated = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
                if correlated:
                    highly_correlated.append((column, correlated))
            
            if highly_correlated:
                validation_results['warnings'].append(
                    f"Found {len(highly_correlated)} pairs of highly correlated features (>0.95)"
                )
        
        # 3. Check for features with too many missing values
        missing_threshold = 0.3  # 30% missing
        high_missing = []
        for col in feature_df.columns:
            missing_pct = feature_df[col].isnull().sum() / len(feature_df)
            if missing_pct > missing_threshold:
                high_missing.append((col, missing_pct))
        
        if high_missing:
            validation_results['issues'].append(
                f"Found {len(high_missing)} features with >{missing_threshold*100}% missing values"
            )
        
        # 4. Check for extreme outliers
        outlier_features = []
        for col in numeric_features.columns:
            if feature_df[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((feature_df[col] - feature_df[col].mean()) / feature_df[col].std())
                extreme_outliers = (z_scores > 5).sum()
                if extreme_outliers > len(feature_df) * 0.01:  # More than 1% extreme outliers
                    outlier_features.append((col, extreme_outliers))
        
        if outlier_features:
            validation_results['warnings'].append(
                f"Found {len(outlier_features)} features with extreme outliers"
            )
        
        # 5. Feature distribution checks
        skewed_features = []
        for col in numeric_features.columns:
            skewness = feature_df[col].skew()
            if abs(skewness) > 2:  # Highly skewed
                skewed_features.append((col, skewness))
        
        if skewed_features:
            validation_results['warnings'].append(
                f"Found {len(skewed_features)} highly skewed features (|skew| > 2)"
            )
        
        # 6. Check data types
        dtype_issues = []
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                # Check if it should be numeric
                try:
                    pd.to_numeric(feature_df[col], errors='coerce')
                    dtype_issues.append(col)
                except:
                    pass
        
        if dtype_issues:
            validation_results['issues'].append(
                f"Found {len(dtype_issues)} features with potential dtype issues"
            )
        
        # 7. Feature value range checks
        for col in numeric_features.columns:
            validation_results['feature_stats'][col] = {
                'min': feature_df[col].min(),
                'max': feature_df[col].max(),
                'mean': feature_df[col].mean(),
                'std': feature_df[col].std(),
                'missing_pct': feature_df[col].isnull().sum() / len(feature_df)
            }
        
        # Display validation summary
        if validation_results['issues']:
            self.console.print("[red]Data Quality Issues Found:[/red]")
            for issue in validation_results['issues']:
                self.console.print(f"  ❌ {issue}")
        
        if validation_results['warnings']:
            self.console.print("[yellow]Data Quality Warnings:[/yellow]")
            for warning in validation_results['warnings']:
                self.console.print(f"  ⚠️  {warning}")
        
        if not validation_results['issues'] and not validation_results['warnings']:
            self.console.print(f"[green]✓ All {feature_type} features passed validation[/green]")
        
        return validation_results

    def auto_fix_features(self, feature_df: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """Automatically fix common feature issues based on validation results."""
        self.console.print("[cyan]Auto-fixing feature issues...[/cyan]")
        
        fixed_df = feature_df.copy()
        fixes_applied = []
        
        # 1. Remove constant features
        constant_cols = []
        for col in fixed_df.select_dtypes(include=[np.number]).columns:
            if fixed_df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            fixed_df = fixed_df.drop(columns=constant_cols)
            fixes_applied.append(f"Removed {len(constant_cols)} constant features")
        
        # 2. Handle highly correlated features
        numeric_features = fixed_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = set()
            for column in upper_triangle.columns:
                correlated = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
                # Keep the first feature, drop the correlated ones
                to_drop.update(correlated)
            
            if to_drop:
                fixed_df = fixed_df.drop(columns=list(to_drop))
                fixes_applied.append(f"Removed {len(to_drop)} highly correlated features")
        
        # 3. Fix skewed features with log transformation
        for col in fixed_df.select_dtypes(include=[np.number]).columns:
            skewness = fixed_df[col].skew()
            if abs(skewness) > 2 and fixed_df[col].min() > 0:  # Only for positive values
                fixed_df[f'{col}_log'] = np.log1p(fixed_df[col])
                fixed_df = fixed_df.drop(columns=[col])
                fixes_applied.append(f"Log-transformed {col} (skewness: {skewness:.2f})")
        
        # 4. Cap extreme outliers
        for col in fixed_df.select_dtypes(include=[np.number]).columns:
            Q1 = fixed_df[col].quantile(0.01)
            Q99 = fixed_df[col].quantile(0.99)
            
            outliers_low = (fixed_df[col] < Q1).sum()
            outliers_high = (fixed_df[col] > Q99).sum()
            
            if outliers_low > 0 or outliers_high > 0:
                fixed_df[col] = fixed_df[col].clip(lower=Q1, upper=Q99)
                fixes_applied.append(f"Capped {outliers_low + outliers_high} outliers in {col}")
        
        # Display fixes summary
        if fixes_applied:
            self.console.print("[green]Applied automatic fixes:[/green]")
            for fix in fixes_applied[:5]:  # Show first 5 fixes
                self.console.print(f"  ✓ {fix}")
            if len(fixes_applied) > 5:
                self.console.print(f"  ... and {len(fixes_applied) - 5} more fixes")
        
        return fixed_df
    
    def estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of a dataframe in MB."""
        return df.memory_usage(deep=True).sum() / 1024**2

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        self.console.print("[cyan]Optimizing data types for memory efficiency...[/cyan]")
        
        start_mem = self.estimate_memory_usage(df)
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Convert string columns with low cardinality to category
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        end_mem = self.estimate_memory_usage(df)
        reduction_pct = (start_mem - end_mem) / start_mem * 100
        
        self.console.print(f"[green]✓ Memory reduced from {start_mem:.2f}MB to {end_mem:.2f}MB ({reduction_pct:.1f}% reduction)[/green]")
        
        return df

    def process_in_chunks(self, df: pd.DataFrame, chunk_size: int = 1000000, 
                        process_func: callable = None, **kwargs) -> pd.DataFrame:
        """Process large dataframes in chunks to avoid memory issues."""
        if len(df) <= chunk_size:
            return process_func(df, **kwargs) if process_func else df
        
        chunks = []
        n_chunks = (len(df) - 1) // chunk_size + 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {n_chunks} chunks...", total=n_chunks)
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                if process_func:
                    chunk = process_func(chunk, **kwargs)
                chunks.append(chunk)
                progress.advance(task)
        
        return pd.concat(chunks, ignore_index=True)

    def create_user_features_optimized(self, ratings_df: pd.DataFrame, 
                                    batch_size: int = 10000) -> pd.DataFrame:
        """Create user features with memory optimization for large datasets."""
        self.console.print("[cyan]Creating user features (memory-optimized)...[/cyan]")
        
        # Check if we need to process in batches
        unique_users = ratings_df['userId'].nunique()
        if unique_users > 100000:
            self.console.print(f"[yellow]Large dataset detected: {unique_users:,} users. Processing in batches...[/yellow]")
        
        # Use categorical userId for memory efficiency
        if ratings_df['userId'].dtype != 'category':
            ratings_df['userId'] = ratings_df['userId'].astype('category')
        
        # Process users in batches
        user_ids = ratings_df['userId'].unique()
        n_batches = (len(user_ids) - 1) // batch_size + 1
        
        user_features_list = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing user batches...", total=n_batches)
            
            for i in range(0, len(user_ids), batch_size):
                batch_users = user_ids[i:i+batch_size]
                batch_ratings = ratings_df[ratings_df['userId'].isin(batch_users)]
                
                # Compute features for this batch
                batch_features = batch_ratings.groupby('userId').agg({
                    'rating': ['count', 'mean', 'std', 'min', 'max'],
                    'movieId': 'nunique',
                    'timestamp': ['min', 'max']
                })
                
                batch_features.columns = ['_'.join(col).strip() for col in batch_features.columns.values]
                
                # Add derived features
                batch_features['rating_std'] = batch_features['rating_std'].fillna(0)
                batch_features['activity_span_days'] = (
                    batch_features['timestamp_max'] - batch_features['timestamp_min']
                ).dt.total_seconds() / (24 * 3600)
                
                batch_features['rating_frequency'] = (
                    batch_features['rating_count'] / (batch_features['activity_span_days'] + 1)
                )
                
                # Optimize memory for this batch
                batch_features = self.optimize_dtypes(batch_features)
                
                user_features_list.append(batch_features)
                progress.advance(task)
                
                # Free memory
                del batch_ratings
                import gc
                gc.collect()
        
        # Combine all batches
        user_features = pd.concat(user_features_list)
        
        # Final optimizations
        user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
        user_features['is_active_user'] = (
            user_features['rating_count'] > user_features['rating_count'].quantile(0.75)
        ).astype('int8')
        user_features['is_picky_user'] = (
            user_features['rating_std'] > user_features['rating_std'].quantile(0.75)
        ).astype('int8')
        
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features for {len(user_features):,} users[/green]")
        self.console.print(f"[green]✓ Memory usage: {self.estimate_memory_usage(user_features):.2f}MB[/green]")
        
        return user_features

    def create_sparse_tfidf_features(self, tags_df: pd.DataFrame, movies_df: pd.DataFrame, 
                                    max_features: int = 100) -> Tuple[sparse.csr_matrix, pd.Index, TfidfVectorizer]:
        """Create sparse TF-IDF features to save memory."""
        self.console.print("[cyan]Creating sparse TF-IDF features...[/cyan]")
        
        # Aggregate tags by movie
        movie_tags = tags_df.groupby('movieId')['tag_clean'].apply(lambda x: ' '.join(x)).reset_index()
        
        # Fill missing movies
        all_movies = pd.DataFrame({'movieId': movies_df['movieId']})
        movie_tags_complete = all_movies.merge(movie_tags, on='movieId', how='left')
        movie_tags_complete['combined_tags'] = movie_tags_complete['combined_tags'].fillna('')
        
        # Create sparse TF-IDF matrix
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7,
            dtype=np.float32  # Use float32 instead of float64
        )
        
        # Process in batches if needed
        if len(movie_tags_complete) > 50000:
            self.console.print("[yellow]Large dataset: Creating TF-IDF in batches...[/yellow]")
            
            # This is a simplified approach - for production, consider using HashingVectorizer
            tfidf_matrix = tfidf.fit_transform(movie_tags_complete['combined_tags'])
        else:
            tfidf_matrix = tfidf.fit_transform(movie_tags_complete['combined_tags'])
        
        self.console.print(f"[green]✓ Created sparse TF-IDF matrix: {tfidf_matrix.shape}, "
                        f"density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}[/green]")
        
        return tfidf_matrix, movie_tags_complete['movieId'], tfidf

    def save_features_efficiently(self, features_dict: Dict[str, pd.DataFrame], 
                                output_dir: Path = PROCESSED_DATA_DIR):
        """Save features using efficient formats."""
        self.console.print("[cyan]Saving features efficiently...[/cyan]")
        
        output_dir.mkdir(exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Saving feature sets...", total=len(features_dict))
            
            for name, df in features_dict.items():
                if isinstance(df, pd.DataFrame):
                    # Use parquet for better compression and faster loading
                    output_path = output_dir / f"{name}.parquet"
                    df.to_parquet(output_path, compression='snappy', index=True)
                    
                    # Log file size
                    file_size_mb = output_path.stat().st_size / 1024**2
                    self.console.print(f"[green]✓ Saved {name}: {file_size_mb:.2f}MB[/green]")
                
                elif sparse.issparse(df):
                    # Save sparse matrices efficiently
                    output_path = output_dir / f"{name}_sparse.npz"
                    sparse.save_npz(output_path, df)
                    
                    file_size_mb = output_path.stat().st_size / 1024**2
                    self.console.print(f"[green]✓ Saved sparse {name}: {file_size_mb:.2f}MB[/green]")
                
                progress.advance(task)

    def validate_memory_usage(self, features_dict: Dict[str, pd.DataFrame], 
                            memory_limit_gb: float = 16.0) -> bool:
        """Validate that total memory usage is within limits."""
        total_memory_mb = 0
        
        table = Table(title="Memory Usage Summary", box=box.ROUNDED)
        table.add_column("Feature Set", style="cyan")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Shape", justify="right")
        
        for name, df in features_dict.items():
            if isinstance(df, pd.DataFrame):
                memory_mb = self.estimate_memory_usage(df)
                total_memory_mb += memory_mb
                table.add_row(name, f"{memory_mb:.2f}", str(df.shape))
            elif sparse.issparse(df):
                # Estimate sparse matrix memory
                memory_mb = (df.data.nbytes + df.indices.nbytes + df.indptr.nbytes) / 1024**2
                total_memory_mb += memory_mb
                table.add_row(f"{name} (sparse)", f"{memory_mb:.2f}", str(df.shape))
        
        table.add_row("[bold]Total", f"[bold]{total_memory_mb:.2f}", "")
        self.console.print(table)
        
        memory_limit_mb = memory_limit_gb * 1024
        if total_memory_mb > memory_limit_mb:
            self.console.print(f"[red]⚠ Warning: Total memory usage ({total_memory_mb:.2f}MB) "
                            f"exceeds limit ({memory_limit_mb:.2f}MB)[/red]")
            return False
        else:
            self.console.print(f"[green]✓ Memory usage within limits ({total_memory_mb:.2f}MB < {memory_limit_mb:.2f}MB)[/green]")
            return True