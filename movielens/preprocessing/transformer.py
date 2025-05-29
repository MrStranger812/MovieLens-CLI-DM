import pandas as pd
import numpy as np
from pandarallel import pandarallel
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
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Initialize pandarallel with 4 workers
pandarallel.initialize(progress_bar=True, nb_workers=4, verbose=1)

class DataTransformer:
    """Advanced data transformation and feature engineering utilities with parallel processing."""
    
    def __init__(self, n_jobs: int = 4):
        self.console = Console()
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.feature_selectors = {}
        self.transformation_log = []
        self.n_jobs = n_jobs
        
        self.console.print(f"[green]✓ DataTransformer initialized with {n_jobs} parallel workers[/green]")
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values with different strategies using parallel processing."""
        self.console.print("[cyan]Handling missing values (parallel)...[/cyan]")
        
        df_clean = df.copy()
        missing_info = []
        
        # Parallel processing for missing value detection
        def check_missing_column(column):
            missing_count = df_clean[column].isnull().sum()
            missing_pct = (missing_count / len(df_clean)) * 100
            
            if missing_count > 0:
                return {
                    'column': column,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct
                }
            return None
        
        missing_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(check_missing_column)(col) for col in df_clean.columns
        )
        
        missing_info = [result for result in missing_results if result is not None]
        
        # Process missing values in parallel
        def process_missing_column(column_info):
            column = column_info['column']
            missing_pct = column_info['missing_percentage']
            
            if strategy == 'auto':
                if missing_pct > 50:
                    return ('drop', column, missing_pct)
                elif df_clean[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                    median_val = df_clean[column].median()
                    return ('fill_median', column, median_val)
                else:
                    mode_val = df_clean[column].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    return ('fill_mode', column, fill_val)
            return None
        
        if missing_info:
            processing_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(process_missing_column)(info) for info in missing_info
            )
            
            # Apply the processing results
            for result in processing_results:
                if result:
                    action, column, value = result
                    if action == 'drop':
                        df_clean = df_clean.drop(columns=[column])
                        self.console.print(f"[yellow]Dropped {column} (>{value:.1f}% missing)[/yellow]")
                    elif action == 'fill_median':
                        df_clean[column] = df_clean[column].fillna(value)
                        self.console.print(f"[green]Filled {column} with median[/green]")
                    elif action == 'fill_mode':
                        df_clean[column] = df_clean[column].fillna(value)
                        self.console.print(f"[green]Filled {column} with mode/Unknown[/green]")
        
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
            
        self.transformation_log.append(f"Missing values handled using {strategy} strategy (parallel)")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate records with parallel processing."""
        self.console.print("[cyan]Checking for duplicates (parallel)...[/cyan]")
        
        initial_count = len(df)
        
        if len(df) > 100000:
            chunk_size = len(df) // self.n_jobs
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            def remove_duplicates_chunk(chunk):
                return chunk.drop_duplicates(subset=subset, keep='first')
            
            processed_chunks = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(remove_duplicates_chunk)(chunk) for chunk in chunks
            )
            
            df_clean = pd.concat(processed_chunks, ignore_index=True)
            df_clean = df_clean.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        if duplicates_removed > 0:
            self.console.print(f"[yellow]Removed {duplicates_removed} duplicate records[/yellow]")
        else:
            self.console.print("[green]✓ No duplicates found[/green]")
            
        self.transformation_log.append(f"Removed {duplicates_removed} duplicate records (parallel)")
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers using parallel processing."""
        self.console.print(f"[cyan]Detecting outliers using {method} method (parallel)...[/cyan]")
        
        df_clean = df.copy()
        
        def process_outliers_column(column):
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
                        return {
                            'column': column,
                            'outlier_count': outlier_count,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
            return None
        
        outlier_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(process_outliers_column)(col) for col in columns
        )
        
        outlier_info = [result for result in outlier_results if result is not None]
        
        for info in outlier_info:
            column = info['column']
            lower_bound = info['lower_bound']
            upper_bound = info['upper_bound']
            
            df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
            df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        
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
        """Create comprehensive user-based features using parallel processing."""
        self.console.print("[cyan]Creating user features (pandarallel)...[/cyan]")
        
        user_features = ratings_df.groupby('userId').parallel_apply(
            lambda x: pd.Series({
                'rating_count': len(x),
                'rating_mean': x['rating'].mean(),
                'rating_std': x['rating'].std(),
                'rating_min': x['rating'].min(),
                'rating_max': x['rating'].max(),
                'movieId_nunique': x['movieId'].nunique(),
                'timestamp_min': x['timestamp'].min(),
                'timestamp_max': x['timestamp'].max()
            })
        )
        
        user_features['rating_std'] = user_features['rating_std'].fillna(0)
        user_features['activity_span_days'] = (
            user_features['timestamp_max'] - user_features['timestamp_min']
        ).dt.total_seconds() / (24 * 3600)
        
        user_features['rating_frequency'] = (
            user_features['rating_count'] / (user_features['activity_span_days'] + 1)
        )
        
        user_features['rating_range'] = user_features['rating_max'] - user_features['rating_min']
        user_features['is_active_user'] = (user_features['rating_count'] > user_features['rating_count'].quantile(0.75)).astype(int)
        user_features['is_picky_user'] = (user_features['rating_std'] > user_features['rating_std'].quantile(0.75)).astype(int)
        
        self.console.print(f"[green]✓ Created {len(user_features.columns)} user features using pandarallel[/green]")
        return user_features
    
    def create_movie_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive movie-based features using parallel processing."""
        self.console.print("[cyan]Creating movie features (pandarallel)...[/cyan]")
        
        movie_features = ratings_df.groupby('movieId').parallel_apply(
            lambda x: pd.Series({
                'rating_count': len(x),
                'rating_mean': x['rating'].mean(),
                'rating_std': x['rating'].std(),
                'rating_min': x['rating'].min(),
                'rating_max': x['rating'].max(),
                'userId_nunique': x['userId'].nunique(),
                'timestamp_min': x['timestamp'].min(),
                'timestamp_max': x['timestamp'].max()
            })
        )
        
        movie_features = movie_features.join(movies_df.set_index('movieId'), how='left')
        
        if 'genre_list' in movie_features.columns:
            movie_features['genre_count'] = movie_features['genre_list'].parallel_apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            all_genres = []
            for genre_list in movie_features['genre_list'].dropna():
                if isinstance(genre_list, list):
                    all_genres.extend(genre_list)
            
            top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
            
            def create_genre_feature(genre):
                if genre != '(no genres listed)':
                    return movie_features['genre_list'].parallel_apply(
                        lambda x: 1 if isinstance(x, list) and genre in x else 0
                    )
                return pd.Series(0, index=movie_features.index)
            
            genre_features = Parallel(n_jobs=self.n_jobs)(
                delayed(create_genre_feature)(genre) for genre in top_genres
            )
            
            for i, genre in enumerate(top_genres):
                if genre != '(no genres listed)':
                    movie_features[f'is_{genre.lower().replace(" ", "_").replace("-", "_")}'] = genre_features[i]
        
        current_year = pd.Timestamp.now().year
        movie_features['movie_age'] = current_year - movie_features['year'].fillna(current_year)
        movie_features['popularity_score'] = (
            movie_features['rating_count'] * movie_features['rating_mean']
        ).fillna(0)
        
        movie_features['rating_std'] = movie_features['rating_std'].fillna(0)
        movie_features['is_popular'] = (
            movie_features['rating_count'] > movie_features['rating_count'].quantile(0.8)
        ).astype(int)
        movie_features['is_highly_rated'] = (
            movie_features['rating_mean'] > movie_features['rating_mean'].quantile(0.8)
        ).astype(int)
        
        self.console.print(f"[green]✓ Created {len(movie_features.columns)} movie features using pandarallel[/green]")
        return movie_features
    
    def create_temporal_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create advanced temporal features using parallel processing."""
        self.console.print("[cyan]Creating temporal features (parallel)...[/cyan]")
        
        temp_df = ratings_df.copy()
        
        if 'timestamp' not in temp_df.columns:
            self.console.print("[red]Error: timestamp column not found in ratings data[/red]")
            return temp_df, pd.DataFrame(), pd.DataFrame()
        
        try:
            user_temporal = temp_df.groupby('userId').parallel_apply(
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
        
        try:
            if 'year' not in temp_df.columns:
                movies_years = ratings_df[['movieId']].drop_duplicates().merge(
                    movies_df[['movieId', 'year']], on='movieId', how='left'
                )
                temp_df = temp_df.merge(movies_years, on='movieId', how='left')
            
            movie_temporal = temp_df.groupby('movieId').parallel_apply(
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
        
        temp_df['season'] = temp_df['timestamp'].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        holidays = [
            ('2010-12-25', '2014-12-25'),
            ('2010-01-01', '2014-01-01'),
            ('2010-07-04', '2014-07-04'),
        ]
        
        temp_df['is_holiday_week'] = 0
        for start, end in holidays:
            holiday_range = pd.date_range(start=start, end=end, freq='YS')
            for holiday in holiday_range:
                mask = (temp_df['timestamp'] >= holiday - pd.Timedelta(days=3)) & \
                    (temp_df['timestamp'] <= holiday + pd.Timedelta(days=3))
                temp_df.loc[mask, 'is_holiday_week'] = 1
        
        self.console.print("[green]✓ Created temporal features using parallel processing[/green]")
        return temp_df, user_temporal, movie_temporal

    def _calculate_rating_trend(self, user_data: pd.DataFrame) -> float:
        """Calculate if user's ratings are trending up or down over time."""
        if len(user_data) < 10:
            return 0.0
        
        sorted_data = user_data.sort_values('timestamp')
        x = np.arange(len(sorted_data))
        y = sorted_data['rating'].values
        
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _calculate_rating_acceleration(self, user_data: pd.DataFrame) -> float:
        """Calculate how fast user's rating frequency is changing."""
        if len(user_data) < 20:
            return 0.0
        
        sorted_data = user_data.sort_values('timestamp')
        monthly_counts = sorted_data.groupby(pd.Grouper(key='timestamp', freq='M')).size()
        
        if len(monthly_counts) < 3:
            return 0.0
        
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
        quarterly_avg = sorted_data.groupby(pd.Grouper(key='timestamp', freq='Q'))['rating'].mean()
        
        if len(quarterly_avg) < 4:
            return 0.0
        
        x = np.arange(len(quarterly_avg))
        y = quarterly_avg.values
        
        log_y = np.log(y + 1e-10)
        slope, _ = np.polyfit(x, log_y, 1)
        
        return -slope
    
    def create_tag_features(self, tags_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Create features from movie tags using parallel processing."""
        self.console.print("[cyan]Creating tag-based features (parallel)...[/cyan]")
        
        movie_tags = tags_df.groupby('movieId')['tag_clean'].parallel_apply(lambda x: ' '.join(x)).reset_index()
        movie_tags.columns = ['movieId', 'combined_tags']
        
        tag_stats = tags_df.groupby('movieId').parallel_apply(
            lambda x: pd.Series({
                'total_tags': len(x),
                'unique_tags': x['tag_clean'].nunique(),
                'users_who_tagged': x['userId'].nunique()
            })
        )
        tag_stats['tag_diversity'] = tag_stats['unique_tags'] / tag_stats['total_tags']
        
        tfidf = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )
        
        all_movies = pd.DataFrame({'movieId': movies_df['movieId']})
        movie_tags_complete = all_movies.merge(movie_tags, on='movieId', how='left')
        movie_tags_complete['combined_tags'] = movie_tags_complete['combined_tags'].fillna('')
        
        tfidf_matrix = tfidf.fit_transform(movie_tags_complete['combined_tags'])
        
        tfidf_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()],
            index=movie_tags_complete['movieId']
        )
        
        positive_words = ['excellent', 'amazing', 'great', 'love', 'wonderful', 'best', 'fantastic', 'masterpiece']
        negative_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'boring', 'waste', 'disappointing']
        
        def calculate_sentiment(tags):
            if pd.isna(tags) or tags == '':
                return pd.Series({'positive_score': 0, 'negative_score': 0, 'sentiment_ratio': 0})
            
            tags_lower = tags.lower()
            positive_count = sum(word in tags_lower for word in positive_words)
            negative_count = sum(word in tags_lower for word in negative_words)
            total = positive_count + negative_count
            
            return pd.Series({
                'positive_score': positive_count,
                'negative_score': negative_count,
                'sentiment_ratio': (positive_count - negative_count) / (total + 1)
            })
        
        sentiment_features = movie_tags_complete['combined_tags'].parallel_apply(calculate_sentiment)
        sentiment_features.index = movie_tags_complete['movieId']
        
        n_topics = 10
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_matrix = lda.fit_transform(tfidf_matrix)
        
        topic_features = pd.DataFrame(
            topic_matrix,
            columns=[f'topic_{i}' for i in range(n_topics)],
            index=movie_tags_complete['movieId']
        )
        
        tag_features = pd.concat([
            tag_stats,
            tfidf_features.iloc[:, :50],
            sentiment_features,
            topic_features
        ], axis=1)
        
        tag_transformers = {
            'tfidf': tfidf,
            'lda': lda
        }
        
        self.console.print(f"[green]✓ Created {tag_features.shape[1]} tag-based features using parallel processing[/green]")
        return tag_features, tag_transformers
    
    def create_user_tag_preferences(self, ratings_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Create user preference profiles using parallel processing."""
        self.console.print("[cyan]Creating user tag preference features (parallel)...[/cyan]")
        
        high_ratings = ratings_df[ratings_df['rating'] >= 4.0][['userId', 'movieId']]
        
        movie_tags_dict = tags_df.groupby('movieId')['tag_clean'].parallel_apply(lambda x: ' '.join(x)).to_dict()
        
        user_ids = high_ratings['userId'].unique()
        batch_size = 10000
        
        user_tfidf = TfidfVectorizer(max_features=50, min_df=2)
        
        sample_size = min(10000, len(high_ratings))
        sample_tags = [movie_tags_dict.get(mid, '') for mid in high_ratings['movieId'].sample(sample_size)]
        user_tfidf.fit(sample_tags)
        
        def process_user_batch(batch_users):
            batch_ratings = high_ratings[high_ratings['userId'].isin(batch_users)]
            
            user_tags = batch_ratings.groupby('userId')['movieId'].apply(
                lambda x: ' '.join([movie_tags_dict.get(mid, '') for mid in x])
            )
            
            batch_tfidf = user_tfidf.transform(user_tags)
            
            batch_features = pd.DataFrame(
                batch_tfidf.toarray(),
                columns=[f'user_pref_{word}' for word in user_tfidf.get_feature_names_out()],
                index=user_tags.index
            )
            
            return batch_features
        
        user_batches = [user_ids[i:i+batch_size] for i in range(0, len(user_ids), batch_size)]
        
        user_tag_features_list = Parallel(n_jobs=self.n_jobs)(
            delayed(process_user_batch)(batch) for batch in user_batches
        )
        
        user_tag_features = pd.concat(user_tag_features_list)
        
        self.console.print(f"[green]✓ Created {user_tag_features.shape[1]} user preference features using parallel processing[/green]")
        return user_tag_features
    
    def create_enhanced_genre_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive genre-based features using parallel processing."""
        self.console.print("[cyan]Creating enhanced genre features (parallel)...[/cyan]")
        
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies_df['genre_list'])
        
        genre_binary = pd.DataFrame(
            genre_matrix,
            columns=[f'genre_{genre}'.lower().replace(' ', '_').replace('-', '_') for genre in mlb.classes_],
            index=movies_df['movieId']
        )
        
        def create_genre_combinations(genre_list):
            if isinstance(genre_list, list):
                return [f"{g1}_{g2}" for i, g1 in enumerate(genre_list) for g2 in genre_list[i+1:]]
            return []
        
        genre_combos = movies_df['genre_list'].parallel_apply(create_genre_combinations)
        
        all_combos = [combo for combos in genre_combos if combos for combo in combos]
        combo_counts = pd.Series(all_combos).value_counts().head(20)
        
        def create_combo_feature(combo):
            g1, g2 = combo.split('_')
            return movies_df['genre_list'].parallel_apply(
                lambda x: 1 if isinstance(x, list) and g1 in x and g2 in x else 0
            )
        
        combo_features = Parallel(n_jobs=self.n_jobs)(
            delayed(create_combo_feature)(combo) for combo in combo_counts.index
        )
        
        for i, combo in enumerate(combo_counts.index):
            genre_binary[f'combo_{combo}'.lower()] = combo_features[i].values
        
        genre_stats = pd.DataFrame(index=movies_df['movieId'])
        
        genre_stats['genre_count'] = movies_df['genre_list'].parallel_apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        genre_stats['is_single_genre'] = (genre_stats['genre_count'] == 1).astype(int)
        genre_stats['is_multi_genre'] = (genre_stats['genre_count'] > 1).astype(int)
        genre_stats['genre_diversity'] = movies_df['genre_list'].parallel_apply(
            lambda x: len(set(x)) / len(x) if isinstance(x, list) and len(x) > 0 else 0
        )
        
        movie_ratings = ratings_df.groupby('movieId').parallel_apply(
            lambda x: pd.Series({
                'avg_rating': x['rating'].mean(),
                'num_ratings': len(x),
                'rating_std': x['rating'].std()
            })
        )
        
        def calculate_genre_performance(genre):
            if genre != '(no genres listed)':
                genre_movies = movies_df[movies_df['genre_list'].apply(
                    lambda x: genre in x if isinstance(x, list) else False
                )]['movieId']
                
                genre_ratings = movie_ratings[movie_ratings.index.isin(genre_movies)]
                
                if len(genre_ratings) > 0:
                    return {
                        'genre': genre,
                        'avg_genre_rating': genre_ratings['avg_rating'].mean(),
                        'genre_popularity': genre_ratings['num_ratings'].sum(),
                        'genre_rating_consistency': genre_ratings['rating_std'].mean()
                    }
            return None
        
        genre_performance_results = Parallel(n_jobs=self.n_jobs)(
            delayed(calculate_genre_performance)(genre) for genre in mlb.classes_
        )
        
        genre_performance = {
            result['genre']: {
                'avg_genre_rating': result['avg_genre_rating'],
                'genre_popularity': result['genre_popularity'],
                'genre_rating_consistency': result['genre_rating_consistency']
            }
            for result in genre_performance_results if result is not None
        }
        
        for col_type in ['avg_genre_rating', 'genre_popularity', 'genre_rating_consistency']:
            genre_stats[f'{col_type}_score'] = movies_df['genre_list'].parallel_apply(
                lambda genres: np.mean([
                    genre_performance.get(g, {}).get(col_type, 0) 
                    for g in genres if isinstance(genres, list) and g in genre_performance
                ]) if isinstance(genres, list) and len(genres) > 0 else 0
            )
        
        genre_counts = pd.Series([g for genres in movies_df['genre_list'] 
                                for g in genres if isinstance(genres, list)]).value_counts()
        
        genre_stats['genre_rarity_score'] = movies_df['genre_list'].parallel_apply(
            lambda genres: np.mean([
                1 / (genre_counts.get(g, 1) + 1) 
                for g in genres if isinstance(genres, list)
            ]) if isinstance(genres, list) and len(genres) > 0 else 0
        )
        
        genre_features = pd.concat([genre_binary, genre_stats], axis=1)
        
        self.console.print(f"[green]✓ Created {genre_features.shape[1]} genre features using parallel processing[/green]")
        return genre_features
    
    def create_cold_start_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                               user_features: pd.DataFrame, movie_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features to handle cold start problem using parallel processing."""
        self.console.print("[cyan]Creating cold start handling features (parallel)...[/cyan]")
        
        user_cold_features = pd.DataFrame(index=user_features.index)
        
        user_rating_counts = ratings_df['userId'].value_counts()
        cold_threshold = user_rating_counts.quantile(0.1)
        
        user_cold_features['is_cold_start'] = (user_features['rating_count'] < cold_threshold).astype(int)
        user_cold_features['rating_confidence'] = 1 - np.exp(-user_features['rating_count'] / 10)
        
        cold_users = user_cold_features[user_cold_features['is_cold_start'] == 1].index
        
        def process_cold_user(user_id):
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            if len(user_ratings) > 0:
                preferred_hour = user_ratings['hour'].mode()[0] if 'hour' in user_ratings.columns else 12
                preferred_day = user_ratings['day_of_week'].mode()[0] if 'day_of_week' in user_ratings.columns else 'Friday'
                return {'user_id': user_id, 'preferred_hour': preferred_hour, 'preferred_day': preferred_day}
            return {'user_id': user_id, 'preferred_hour': 12, 'preferred_day': 'Friday'}
        
        if len(cold_users) > 0:
            cold_user_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_cold_user)(user_id) for user_id in cold_users
            )
            
            for result in cold_user_results:
                user_cold_features.loc[result['user_id'], 'preferred_hour'] = result['preferred_hour']
                user_cold_features.loc[result['user_id'], 'preferred_day'] = result['preferred_day']
        
        time_features = ['rating_count', 'activity_span_days', 'rating_frequency']
        available_time_features = [f for f in time_features if f in user_features.columns]
        
        if available_time_features:
            scaler = StandardScaler()
            user_time_scaled = scaler.fit_transform(user_features[available_time_features].fillna(0))
            
            n_clusters = min(20, len(user_features) // 100)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            user_cold_features['user_cluster'] = kmeans.fit_predict(user_time_scaled)
            
            cluster_stats = ratings_df.merge(
                user_cold_features[['user_cluster']], 
                left_on='userId', 
                right_index=True
            ).groupby('user_cluster')['rating'].agg(['mean', 'std'])
            
            user_cold_features['cluster_avg_rating'] = user_cold_features['user_cluster'].map(cluster_stats['mean'])
            user_cold_features['cluster_rating_std'] = user_cold_features['user_cluster'].map(cluster_stats['std'])
        
        movie_cold_features = pd.DataFrame(index=movie_features.index)
        
        movie_rating_counts = ratings_df['movieId'].value_counts()
        movie_cold_threshold = movie_rating_counts.quantile(0.1)
        
        movie_cold_features['is_cold_start'] = (movie_features['rating_count'] < movie_cold_threshold).astype(int)
        movie_cold_features['rating_confidence'] = 1 - np.exp(-movie_features['rating_count'] / 10)
        
        if 'genres' in movies_df.columns:
            def calculate_genre_rating(genre):
                if genre and genre != '(no genres listed)':
                    genre_movies = movies_df[movies_df['genres'].str.contains(genre, na=False)]['movieId']
                    genre_ratings = ratings_df[ratings_df['movieId'].isin(genre_movies)]
                    if len(genre_ratings) > 0:
                        return {'genre': genre, 'avg_rating': genre_ratings['rating'].mean()}
                return {'genre': genre, 'avg_rating': 3.5}
            
            unique_genres = movies_df['genres'].str.split('|').explode().unique()
            genre_rating_results = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_genre_rating)(genre) for genre in unique_genres
            )
            
            genre_avg_ratings = {result['genre']: result['avg_rating'] for result in genre_rating_results}
            
            def calculate_expected_rating(genres_str):
                if pd.notna(genres_str):
                    genres = genres_str.split('|')
                    valid_ratings = [genre_avg_ratings.get(g, 3.5) for g in genres if g in genre_avg_ratings]
                    return np.mean(valid_ratings) if valid_ratings else 3.5
                return 3.5
            
            genre_expected_ratings = movies_df.set_index('movieId')['genres'].parallel_apply(calculate_expected_rating)
            movie_cold_features['genre_expected_rating'] = genre_expected_ratings.reindex(movie_cold_features.index, fill_value=3.5)
        
        content_features = [col for col in movie_features.columns 
                        if col.startswith('genre_') or col.startswith('is_')]
        
        if content_features:
            movie_content_scaled = StandardScaler().fit_transform(
                movie_features[content_features].fillna(0)
            )
            
            movie_kmeans = KMeans(n_clusters=min(50, len(movie_features) // 100), random_state=42)
            movie_cold_features['movie_cluster'] = movie_kmeans.fit_predict(movie_content_scaled)
            
            movie_cluster_stats = ratings_df.merge(
                movie_cold_features[['movie_cluster']], 
                left_on='movieId', 
                right_index=True
            ).groupby('movie_cluster')['rating'].agg(['mean', 'std', 'count'])
            
            movie_cold_features['cluster_avg_rating'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['mean'])
            movie_cold_features['cluster_rating_std'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['std'])
            movie_cold_features['cluster_popularity'] = movie_cold_features['movie_cluster'].map(movie_cluster_stats['count'])
        
        global_avg_rating = ratings_df['rating'].mean()
        user_cold_features['global_avg_rating'] = global_avg_rating
        movie_cold_features['global_avg_rating'] = global_avg_rating
        
        self.console.print(f"[green]✓ Created {user_cold_features.shape[1]} user cold start features using parallel processing[/green]")
        self.console.print(f"[green]✓ Created {movie_cold_features.shape[1]} movie cold start features using parallel processing[/green]")
        return user_cold_features, movie_cold_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str], 
                                  method: str = 'label') -> pd.DataFrame:
        """Encode categorical features using parallel processing."""
        self.console.print(f"[cyan]Encoding categorical features using {method} encoding (parallel)...[/cyan]")
        
        df_encoded = df.copy()
        
        def encode_column(column):
            if column in df_encoded.columns:
                if method == 'label':
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(df_encoded[column].astype(str))
                    return column, encoded_values, le
                elif method == 'onehot':
                    dummy_df = pd.get_dummies(df_encoded[column], prefix=column)
                    return column, dummy_df, None
            return None
        
        encoding_results = Parallel(n_jobs=self.n_jobs)(
            delayed(encode_column)(col) for col in categorical_columns
        )
        
        for result in encoding_results:
            if result:
                column, encoded_data, encoder = result
                if method == 'label':
                    df_encoded[f'{column}_encoded'] = encoded_data
                    self.encoders[f'{column}_label'] = encoder
                elif method == 'onehot':
                    df_encoded = pd.concat([df_encoded, encoded_data], axis=1)
                
                self.console.print(f"[green]✓ Encoded {column}[/green]")
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, numerical_columns: List[str], 
                          method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features using parallel processing."""
        self.console.print(f"[cyan]Normalizing features using {method} scaling (parallel)...[/cyan]")
        
        df_normalized = df.copy()
        
        def normalize_column(column):
            if column in df_normalized.columns:
                if method == 'standard':
                    scaler = StandardScaler()
                    normalized_values = scaler.fit_transform(df_normalized[[column]]).flatten()
                    return column, normalized_values, scaler, 'standard'
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                    normalized_values = scaler.fit_transform(df_normalized[[column]]).flatten()
                    return column, normalized_values, scaler, 'minmax'
                elif method == 'robust':
                    median = df_normalized[column].median()
                    q75 = df_normalized[column].quantile(0.75)
                    q25 = df_normalized[column].quantile(0.25)
                    iqr = q75 - q25
                    
                    if iqr > 0:
                        normalized_values = (df_normalized[column] - median) / iqr
                    else:
                        normalized_values = pd.Series(0, index=df_normalized.index)
                    return column, normalized_values, None, 'robust'
            return None
        
        normalization_results = Parallel(n_jobs=self.n_jobs)(
            delayed(normalize_column)(col) for col in numerical_columns
        )
        
        for result in normalization_results:
            if result:
                column, normalized_data, scaler, method_used = result
                df_normalized[f'{column}_scaled'] = normalized_data
                if scaler:
                    self.scalers[f'{column}_{method_used}'] = scaler
                
                self.console.print(f"[green]✓ Normalized {column}[/green]")
        
        return df_normalized
    
    def apply_pca(self, df: pd.DataFrame, feature_columns: List[str], 
                  n_components: Optional[int] = None, variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply PCA for dimensionality reduction with parallel data preparation."""
        self.console.print("[cyan]Applying PCA for dimensionality reduction (parallel data prep)...[/cyan]")
        
        def check_numeric_column(col):
            if col in df.columns:
                return col, df[col].select_dtypes(include=[np.number]).fillna(0)
            return None
        
        numeric_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_numeric_column)(col) for col in feature_columns
        )
        
        numeric_columns = {result[0]: result[1] for result in numeric_results if result is not None}
        feature_data = pd.concat(numeric_columns.values(), axis=1)
        
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(feature_data)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(feature_data)
        
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
        
        self.pca_models['main_pca'] = pca
        
        explained_variance = pd.DataFrame({
            'Component': pca_columns,
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        
        self.console.print(f"[green]✓ Reduced {len(feature_columns)} features to {n_components} components[/green]")
        self.console.print(f"[green]✓ Explained variance: {pca.explained_variance_ratio_.sum():.3f}[/green]")
        return pca_df, explained_variance
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features using parallel processing."""
        self.console.print("[cyan]Creating interaction features (parallel)...[/cyan]")
        
        df_interactions = df.copy()
        
        def create_interaction_pair(feature_pair):
            feature1, feature2 = feature_pair
            if feature1 in df_interactions.columns and feature2 in df_interactions.columns:
                interactions = {}
                
                interaction_name = f'{feature1}_x_{feature2}'
                interactions[interaction_name] = df_interactions[feature1] * df_interactions[feature2]
                
                if (df_interactions[feature2] != 0).all():
                    ratio_name = f'{feature1}_div_{feature2}'
                    interactions[ratio_name] = df_interactions[feature1] / df_interactions[feature2]
                
                return interactions
            return {}
        
        interaction_results = Parallel(n_jobs=self.n_jobs)(
            delayed(create_interaction_pair)(pair) for pair in feature_pairs
        )
        
        created_features = []
        for interactions in interaction_results:
            for name, values in interactions.items():
                df_interactions[name] = values
                created_features.append(name)
        
        self.console.print(f"[green]✓ Created {len(created_features)} interaction features using parallel processing[/green]")
        return df_interactions
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 'auto', method: str = 'univariate') -> pd.DataFrame:
        """Select the most important features."""
        self.console.print(f"[cyan]Selecting features using {method} method...[/cyan]")
        
        if k == 'auto':
            k = min(50, X.shape[1] // 2)
        
        if method == 'univariate':
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.feature_selectors['univariate'] = selector
        
        self.console.print(f"[green]✓ Selected {len(selected_features)} most important features[/green]")
        return X_selected_df
    
    def create_user_item_matrix(self, ratings_df: pd.DataFrame, sparse: bool = True) -> Tuple[np.ndarray, Dict, Dict]:
        """Create user-item matrix using parallel processing."""
        self.console.print("[cyan]Creating user-item matrix (parallel)...[/cyan]")
        
        users = sorted(ratings_df['userId'].unique())
        movies = sorted(ratings_df['movieId'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(movies)}
        
        if sparse:
            from scipy.sparse import coo_matrix
            
            user_indices = ratings_df['userId'].map(user_to_idx).values
            movie_indices = ratings_df['movieId'].map(movie_to_idx).values
            ratings_values = ratings_df['rating'].values
            
            matrix = coo_matrix(
                (ratings_values, (user_indices, movie_indices)),
                shape=(len(users), len(movies)),
                dtype=np.float32
            ).tocsr()
            
            total_elements = matrix.shape[0] * matrix.shape[1]
            sparsity = 1 - (matrix.nnz / total_elements)
        else:
            matrix = np.zeros((len(users), len(movies)), dtype=np.float32)
            
            chunk_size = len(ratings_df) // self.n_jobs
            
            def process_chunk(start_idx, end_idx):
                chunk = ratings_df.iloc[start_idx:end_idx]
                chunk_matrix = np.zeros((len(users), len(movies)), dtype=np.float32)
                
                for _, row in chunk.iterrows():
                    user_idx = user_to_idx[row['userId']]
                    movie_idx = movie_to_idx[row['movieId']]
                    chunk_matrix[user_idx, movie_idx] = row['rating']
                
                return chunk_matrix
            
            chunk_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_chunk)(i, min(i + chunk_size, len(ratings_df))) 
                for i in range(0, len(ratings_df), chunk_size)
            )
            
            for chunk_matrix in chunk_results:
                matrix += chunk_matrix
            
            sparsity = 1 - (np.count_nonzero(matrix) / matrix.size)
        
        self.console.print(f"[green]✓ Created {matrix.shape[0]}x{matrix.shape[1]} user-item matrix using parallel processing[/green]")
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
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types using parallel processing."""
        self.console.print("[cyan]Optimizing data types for memory efficiency (parallel)...[/cyan]")
        
        start_mem = self.estimate_memory_usage(df)
        df_optimized = df.copy()
        
        def optimize_numeric_column(col):
            if col in df_optimized.columns:
                if df_optimized[col].dtype == 'int64':
                    col_min = df_optimized[col].min()
                    col_max = df_optimized[col].max()
                    
                    if col_min >= -128 and col_max <= 127:
                        return col, 'int8'
                    elif col_min >= -32768 and col_max <= 32767:
                        return col, 'int16'
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        return col, 'int32'
                elif df_optimized[col].dtype == 'float64':
                    return col, 'float32'
                elif df_optimized[col].dtype == 'object':
                    num_unique = df_optimized[col].nunique()
                    num_total = len(df_optimized[col])
                    if num_unique / num_total < 0.5:
                        return col, 'category'
            return None
        
        optimization_results = Parallel(n_jobs=self.n_jobs)(
            delayed(optimize_numeric_column)(col) for col in df_optimized.columns
        )
        
        for result in optimization_results:
            if result:
                col, new_dtype = result
                df_optimized[col] = df_optimized[col].astype(new_dtype)
        
        end_mem = self.estimate_memory_usage(df_optimized)
        reduction_pct = (start_mem - end_mem) / start_mem * 100
        
        self.console.print(f"[green]✓ Memory reduced from {start_mem:.2f}MB to {end_mem:.2f}MB ({reduction_pct:.1f}% reduction) using parallel processing[/green]")
        return df_optimized
    
    def estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of a dataframe in MB."""
        return df.memory_usage(deep=True).sum() / 1024**2
    
    def process_in_chunks(self, df: pd.DataFrame, chunk_size: int = 1000000, 
                        process_func: callable = None, **kwargs) -> pd.DataFrame:
        """Process large dataframes in chunks using parallel processing."""
        if len(df) <= chunk_size:
            return process_func(df, **kwargs) if process_func else df
        
        chunks = []
        n_chunks = (len(df) - 1) // chunk_size + 1
        
        def process_chunk(chunk):
            if process_func:
                return process_func(chunk, **kwargs)
            return chunk
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {n_chunks} chunks...", total=n_chunks)
            
            chunk_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_chunk)(df.iloc[i:i+chunk_size]) 
                for i in range(0, len(df), chunk_size)
            )
            
            for result in chunk_results:
                chunks.append(result)
                progress.advance(task)
        
        return pd.concat(chunks, ignore_index=True)

    def create_user_features_optimized(self, ratings_df: pd.DataFrame, 
                                    batch_size: int = 10000) -> pd.DataFrame:
        """Create user features with memory optimization for large datasets using parallel processing."""
        self.console.print("[cyan]Creating user features (memory-optimized, parallel)...[/cyan]")
        
        unique_users = ratings_df['userId'].nunique()
        if unique_users > 100000:
            self.console.print(f"[yellow]Large dataset detected: {unique_users:,} users. Processing in batches...[/yellow]")
        
        if ratings_df['userId'].dtype != 'category':
            ratings_df['userId'] = ratings_df['userId'].astype('category')
        
        user_ids = ratings_df['userId'].unique()
        n_batches = (len(user_ids) - 1) // batch_size + 1
        
        user_features_list = []
        
        def process_user_batch(batch_users):
            batch_ratings = ratings_df[ratings_df['userId'].isin(batch_users)]
            
            batch_features = batch_ratings.groupby('userId').parallel_apply(
                lambda x: pd.Series({
                    'rating_count': len(x),
                    'rating_mean': x['rating'].mean(),
                    'rating_std': x['rating'].std(),
                    'rating_min': x['rating'].min(),
                    'rating_max': x['rating'].max(),
                    'movieId_nunique': x['movieId'].nunique(),
                    'timestamp_min': x['timestamp'].min(),
                    'timestamp_max': x['timestamp'].max()
                })
            )
            
            batch_features['rating_std'] = batch_features['rating_std'].fillna(0)
            batch_features['activity_span_days'] = (
                batch_features['timestamp_max'] - batch_features['timestamp_min']
            ).dt.total_seconds() / (24 * 3600)
            
            batch_features['rating_frequency'] = (
                batch_features['rating_count'] / (batch_features['activity_span_days'] + 1)
            )
            
            return self.optimize_dtypes(batch_features)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing user batches...", total=n_batches)
            
            batch_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_user_batch)(user_ids[i:i+batch_size]) 
                for i in range(0, len(user_ids), batch_size)
            )
            
            for result in batch_results:
                user_features_list.append(result)
                progress.advance(task)
        
        user_features = pd.concat(user_features_list)
        
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
        """Create sparse TF-IDF features to save memory using parallel processing."""
        self.console.print("[cyan]Creating sparse TF-IDF features (parallel)...[/cyan]")
        
        movie_tags = tags_df.groupby('movieId')['tag_clean'].parallel_apply(lambda x: ' '.join(x)).reset_index()
        
        all_movies = pd.DataFrame({'movieId': movies_df['movieId']})
        movie_tags_complete = all_movies.merge(movie_tags, on='movieId', how='left')
        movie_tags_complete['combined_tags'] = movie_tags_complete['combined_tags'].fillna('')
        
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7,
            dtype=np.float32
        )
        
        if len(movie_tags_complete) > 50000:
            self.console.print("[yellow]Large dataset: Creating TF-IDF in batches...[/yellow]")
            chunk_size = len(movie_tags_complete) // self.n_jobs
            
            def process_tfidf_chunk(start_idx, end_idx):
                chunk = movie_tags_complete['combined_tags'].iloc[start_idx:end_idx]
                return tfidf.fit_transform(chunk) if start_idx == 0 else tfidf.transform(chunk)
            
            tfidf_chunks = Parallel(n_jobs=self.n_jobs)(
                delayed(process_tfidf_chunk)(i, min(i + chunk_size, len(movie_tags_complete))) 
                for i in range(0, len(movie_tags_complete), chunk_size)
            )
            
            tfidf_matrix = sparse.vstack(tfidf_chunks)
        else:
            tfidf_matrix = tfidf.fit_transform(movie_tags_complete['combined_tags'])
        
        self.console.print(f"[green]✓ Created sparse TF-IDF matrix: {tfidf_matrix.shape}, "
                        f"density: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}[/green]")
        return tfidf_matrix, movie_tags_complete['movieId'], tfidf

    def save_features_efficiently(self, features_dict: Dict[str, pd.DataFrame], 
                                output_dir: Path = PROCESSED_DATA_DIR):
        """Save features using efficient formats with parallel processing."""
        self.console.print("[cyan]Saving features efficiently (parallel)...[/cyan]")
        
        output_dir.mkdir(exist_ok=True)
        
        def save_feature(name, data):
            if isinstance(data, pd.DataFrame):
                output_path = output_dir / f"{name}.parquet"
                data.to_parquet(output_path, compression='snappy', index=True)
                file_size_mb = output_path.stat().st_size / 1024**2
                return f"[green]✓ Saved {name}: {file_size_mb:.2f}MB[/green]"
            elif sparse.issparse(data):
                output_path = output_dir / f"{name}_sparse.npz"
                sparse.save_npz(output_path, data)
                file_size_mb = output_path.stat().st_size / 1024**2
                return f"[green]✓ Saved sparse {name}: {file_size_mb:.2f}MB[/green]"
            return None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Saving feature sets...", total=len(features_dict))
            
            save_results = Parallel(n_jobs=self.n_jobs)(
                delayed(save_feature)(name, data) for name, data in features_dict.items()
            )
            
            for result in save_results:
                if result:
                    self.console.print(result)
                progress.advance(task)

    def validate_memory_usage(self, features_dict: Dict[str, pd.DataFrame], 
                            memory_limit_gb: float = 16.0) -> bool:
        """Validate that total memory usage is within limits using parallel processing."""
        total_memory_mb = 0
        
        def estimate_feature_memory(name, data):
            if isinstance(data, pd.DataFrame):
                memory_mb = self.estimate_memory_usage(data)
                return name, memory_mb, data.shape
            elif sparse.issparse(data):
                memory_mb = (data.data.nbytes + data.indices.nbytes + data.indptr.nbytes) / 1024**2
                return f"{name} (sparse)", memory_mb, data.shape
            return None
        
        memory_results = Parallel(n_jobs=self.n_jobs)(
            delayed(estimate_feature_memory)(name, data) for name, data in features_dict.items()
        )
        
        table = Table(title="Memory Usage Summary", box=box.ROUNDED)
        table.add_column("Feature Set", style="cyan")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Shape", justify="right")
        
        for result in memory_results:
            if result:
                name, memory_mb, shape = result
                total_memory_mb += memory_mb
                table.add_row(name, f"{memory_mb:.2f}", str(shape))
        
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

    def validate_features(self, feature_df: pd.DataFrame, feature_type: str = "general") -> Dict:
        """Validate feature quality and identify potential issues using parallel processing."""
        self.console.print(f"[cyan]Validating {feature_type} features (parallel)...[/cyan]")
        
        validation_results = {
            'total_features': len(feature_df.columns),
            'issues': [],
            'warnings': [],
            'feature_stats': {}
        }
        
        def check_constant_feature(col):
            if feature_df[col].nunique() == 1 and col in feature_df.select_dtypes(include=[np.number]).columns:
                return col
            return None
        
        constant_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_constant_feature)(col) for col in feature_df.columns
        )
        constant_features = [col for col in constant_results if col is not None]
        
        if constant_features:
            validation_results['issues'].append(f"Found {len(constant_features)} constant features: {constant_features[:5]}")
        
        numeric_features = feature_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            def check_correlation(column):
                correlated = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
                if correlated:
                    return (column, correlated)
                return None
            
            corr_results = Parallel(n_jobs=self.n_jobs)(
                delayed(check_correlation)(col) for col in upper_triangle.columns
            )
            highly_correlated = [res for res in corr_results if res is not None]
            
            if highly_correlated:
                validation_results['warnings'].append(
                    f"Found {len(highly_correlated)} pairs of highly correlated features (>0.95)"
                )
        
        missing_threshold = 0.3
        def check_missing(col):
            missing_pct = feature_df[col].isnull().sum() / len(feature_df)
            if missing_pct > missing_threshold:
                return (col, missing_pct)
            return None
        
        missing_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_missing)(col) for col in feature_df.columns
        )
        high_missing = [res for res in missing_results if res is not None]
        
        if high_missing:
            validation_results['issues'].append(
                f"Found {len(high_missing)} features with >{missing_threshold*100}% missing values"
            )
        
        def check_outliers(col):
            if col in numeric_features.columns and feature_df[col].std() > 0:
                z_scores = np.abs((feature_df[col] - feature_df[col].mean()) / feature_df[col].std())
                extreme_outliers = (z_scores > 5).sum()
                if extreme_outliers > len(feature_df) * 0.01:
                    return (col, extreme_outliers)
            return None
        
        outlier_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_outliers)(col) for col in numeric_features.columns
        )
        outlier_features = [res for res in outlier_results if res is not None]
        
        if outlier_features:
            validation_results['warnings'].append(
                f"Found {len(outlier_features)} features with extreme outliers"
            )
        
        def check_skewness(col):
            skewness = feature_df[col].skew()
            if abs(skewness) > 2:
                return (col, skewness)
            return None
        
        skew_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_skewness)(col) for col in numeric_features.columns
        )
        skewed_features = [res for res in skew_results if res is not None]
        
        if skewed_features:
            validation_results['warnings'].append(
                f"Found {len(skewed_features)} highly skewed features (|skew| > 2)"
            )
        
        def check_dtype(col):
            if feature_df[col].dtype == 'object':
                try:
                    pd.to_numeric(feature_df[col], errors='coerce')
                    return col
                except:
                    pass
            return None
        
        dtype_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_dtype)(col) for col in feature_df.columns
        )
        dtype_issues = [col for col in dtype_results if col is not None]
        
        if dtype_issues:
            validation_results['issues'].append(
                f"Found {len(dtype_issues)} features with potential dtype issues"
            )
        
        def compute_stats(col):
            if col in numeric_features.columns:
                return col, {
                    'min': feature_df[col].min(),
                    'max': feature_df[col].max(),
                    'mean': feature_df[col].mean(),
                    'std': feature_df[col].std(),
                    'missing_pct': feature_df[col].isnull().sum() / len(feature_df)
                }
            return None
        
        stats_results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_stats)(col) for col in numeric_features.columns
        )
        for result in stats_results:
            if result:
                col, stats = result
                validation_results['feature_stats'][col] = stats
        
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
        """Automatically fix common feature issues based on validation results using parallel processing."""
        self.console.print("[cyan]Auto-fixing feature issues (parallel)...[/cyan]")
        
        fixed_df = feature_df.copy()
        fixes_applied = []
        
        def check_constant(col):
            if fixed_df[col].nunique() == 1 and col in fixed_df.select_dtypes(include=[np.number]).columns:
                return col
            return None
        
        constant_results = Parallel(n_jobs=self.n_jobs)(
            delayed(check_constant)(col) for col in fixed_df.columns
        )
        constant_cols = [col for col in constant_results if col is not None]
        
        if constant_cols:
            fixed_df = fixed_df.drop(columns=constant_cols)
            fixes_applied.append(f"Removed {len(constant_cols)} constant features")
        
        numeric_features = fixed_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            def find_correlated(column):
                correlated = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
                return correlated if correlated else None
            
            corr_results = Parallel(n_jobs=self.n_jobs)(
                delayed(find_correlated)(col) for col in upper_triangle.columns
            )
            to_drop = set()
            for correlated in corr_results:
                if correlated:
                    to_drop.update(correlated)
            
            if to_drop:
                fixed_df = fixed_df.drop(columns=list(to_drop))
                fixes_applied.append(f"Removed {len(to_drop)} highly correlated features")
        
        def fix_skewness(col):
            skewness = fixed_df[col].skew()
            if abs(skewness) > 2 and fixed_df[col].min() > 0:
                return col, np.log1p(fixed_df[col]), skewness
            return None
        
        skew_results = Parallel(n_jobs=self.n_jobs)(
            delayed(fix_skewness)(col) for col in fixed_df.select_dtypes(include=[np.number]).columns
        )
        
        for result in skew_results:
            if result:
                col, log_values, skewness = result
                fixed_df[f'{col}_log'] = log_values
                fixed_df = fixed_df.drop(columns=[col])
                fixes_applied.append(f"Log-transformed {col} (skewness: {skewness:.2f})")
        
        def cap_outliers(col):
            Q1 = fixed_df[col].quantile(0.01)
            Q99 = fixed_df[col].quantile(0.99)
            outliers_low = (fixed_df[col] < Q1).sum()
            outliers_high = (fixed_df[col] > Q99).sum()
            if outliers_low > 0 or outliers_high > 0:
                return col, fixed_df[col].clip(lower=Q1, upper=Q99), outliers_low + outliers_high
            return None
        
        outlier_results = Parallel(n_jobs=self.n_jobs)(
            delayed(cap_outliers)(col) for col in fixed_df.select_dtypes(include=[np.number]).columns
        )
        
        for result in outlier_results:
            if result:
                col, capped_values, outlier_count = result
                fixed_df[col] = capped_values
                fixes_applied.append(f"Capped {outlier_count} outliers in {col}")
        
        if fixes_applied:
            self.console.print("[green]Applied automatic fixes:[/green]")
            for fix in fixes_applied[:5]:
                self.console.print(f"  ✓ {fix}")
            if len(fixes_applied) > 5:
                self.console.print(f"  ... and {len(fixes_applied) - 5} more fixes")
        
        return fixed_df