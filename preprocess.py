import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
import nltk
import emoji
import holidays
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import spacy
import logging
import shutil
import sys
import glob

# Download resources if needed
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
try:
    spacy_nlp = spacy.load('en_core_web_sm')  # Spacy model
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    spacy_nlp = spacy.load('en_core_web_sm')

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stop_words = set(nltk.corpus.stopwords.words('english'))

# === Custom Transformers ===
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clean = X.copy()
        for col in self.text_columns:
            X_clean[col] = (
                X_clean[col]
                .astype(str)
                .str.lower()
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(r'\d+', '', regex=True)
                .apply(lambda x: emoji.replace_emoji(x, replace=''))
                .apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
            )
        return X_clean

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def get_sentiment(self, text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def get_ner_counts(self, text):
        try:
            doc = spacy_nlp(text)
            return len(doc.ents)
        except:
            return 0

    def transform(self, X):
        features = pd.DataFrame()
        for col in self.text_columns:
            features[f'{col}_char_count'] = X[col].apply(len)
            features[f'{col}_word_count'] = X[col].apply(lambda x: len(x.split()))
            features[f'{col}_unique_word_ratio'] = X[col].apply(
                lambda x: len(set(x.split())) / (len(x.split()) + 1e-5))
            features[f'{col}_sentiment'] = X[col].apply(self.get_sentiment)
            features[f'{col}_ner_count'] = X[col].apply(self.get_ner_counts)
        return features

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column
        self.holidays = holidays.CountryHoliday('US')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df['hour'] = df[self.date_column].dt.hour
        df['dayofweek'] = df[self.date_column].dt.dayofweek
        df['month'] = df[self.date_column].dt.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['quarter'] = df[self.date_column].dt.quarter
        df['is_holiday'] = df[self.date_column].apply(lambda x: int(x in self.holidays))
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df[['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend', 'quarter', 'is_holiday']]

class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert pandas Series to list to avoid index issues
        if hasattr(X, 'tolist'):
            X = X.tolist()
        elif hasattr(X, 'values'):
            X = X.values.tolist()
        return self.model.encode(X, show_progress_bar=False)

# === Data Functions ===
def load_data(file_path):
    """
    Load data from either a single file or directory.
    If directory, merge all CSV files and add category column based on filename.
    """
    if os.path.isfile(file_path):
        # Single file case
        logger.info(f"Loading data from file: {file_path}")
        return pd.read_csv(file_path)
    
    elif os.path.isdir(file_path):
        # Directory case - merge all CSV files
        logger.info(f"Loading data from directory: {file_path}")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(file_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {file_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        
        # Load and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add category column based on filename (without extension)
                category = os.path.splitext(os.path.basename(csv_file))[0]
                df['category'] = category
                
                logger.info(f"Loaded {len(df)} rows from {os.path.basename(csv_file)} (category: {category})")
                dataframes.append(df)
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError(f"No valid CSV files could be loaded from directory: {file_path}")
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Categories found: {merged_df['category'].value_counts().to_dict()}")
        
        return merged_df
    
    else:
        raise ValueError(f"Path does not exist: {file_path}")

def save_processed_data(X_train, X_test, X_eval, y_train, y_test, y_eval, preprocessor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sparse.save_npz(os.path.join(output_dir, 'X_train.npz'), sparse.csr_matrix(X_train))
    sparse.save_npz(os.path.join(output_dir, 'X_test.npz'), sparse.csr_matrix(X_test))
    sparse.save_npz(os.path.join(output_dir, 'X_eval.npz'), sparse.csr_matrix(X_eval))
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'y_eval.npy'), y_eval)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    logger.info(f"Saved processed data to {output_dir}")

def detect_datetime_column(df):
    """Automatically detect datetime column from common patterns"""
    datetime_patterns = [
        'date_published', 'date', 'datetime', 'timestamp', 'created_at', 
        'published_at', 'time', 'publish_date', 'created', 'published'
    ]
    
    # First try exact matches (case insensitive)
    for col in df.columns:
        if col.lower() in [p.lower() for p in datetime_patterns]:
            logger.info(f"Found datetime column: {col}")
            return col
    
    # Then try partial matches
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['date', 'time']):
            logger.info(f"Found potential datetime column: {col}")
            return col
    
    # If no matches, check for datetime-like data types
    for col in df.columns:
        try:
            # Try to convert first few non-null values to datetime
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                pd.to_datetime(sample, errors='raise')
                logger.info(f"Found datetime column by type detection: {col}")
                return col
        except:
            continue
    
    return None

def time_based_split(df, datetime_col, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    """Split data chronologically based on datetime column"""
    if datetime_col not in df.columns:
        raise KeyError(f"Column '{datetime_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    df = df.sort_values(datetime_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    eval_end = train_end + int(n * eval_ratio)
    
    train_df = df.iloc[:train_end]
    eval_df = df.iloc[train_end:eval_end]
    test_df = df.iloc[eval_end:]
    
    logger.info(f"Split sizes - Train: {len(train_df)} ({len(train_df)/n:.1%}), "
                f"Eval: {len(eval_df)} ({len(eval_df)/n:.1%}), "
                f"Test: {len(test_df)} ({len(test_df)/n:.1%})")
    return train_df, eval_df, test_df

# === Preprocessing Pipeline ===
def preprocess_data(input_path, output_dir="processed_data", 
                   train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15, 
                   datetime_col=None, target_col=None, text_cols=None, cat_cols=None):
    # Validate ratios
    if abs((train_ratio + eval_ratio + test_ratio) - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")

    # Load data (handles both files and directories)
    full_df = load_data(input_path)
    logger.info(f"Loaded data with shape: {full_df.shape}")
    logger.info(f"Columns: {list(full_df.columns)}")
    
    # Auto-detect datetime column if not provided
    if datetime_col is None:
        datetime_col = detect_datetime_column(full_df)
        if datetime_col is None:
            raise ValueError("No datetime column found. Please specify one using --datetime_col parameter")
    
    # Auto-detect target column if not provided
    if target_col is None:
        target_patterns = ['ctr', 'click_through_rate', 'url_ctr', 'target', 'y']
        for col in full_df.columns:
            if col.lower().replace(' ', '_') in [p.lower() for p in target_patterns]:
                target_col = col
                logger.info(f"Found target column: {target_col}")
                break
        if target_col is None:
            raise ValueError("No target column found. Please specify one using --target_col parameter")
    
    # Auto-detect text columns if not provided
    if text_cols is None:
        text_patterns = ['title', 'subtitle', 'description', 'content', 'text']
        text_cols = []
        for col in full_df.columns:
            if col.lower() in text_patterns:
                text_cols.append(col)
        if not text_cols:
            logger.warning("No text columns found, using default patterns")
            text_cols = ['title', 'subtitle']  # fallback
    
    # Auto-detect categorical columns if not provided
    if cat_cols is None:
        cat_cols = []
        for col in full_df.columns:
            if (col not in [datetime_col, target_col] + text_cols and 
                (full_df[col].dtype == 'object' or full_df[col].nunique() < 50)):
                cat_cols.append(col)
        
        # Always include 'category' column if it exists (from directory loading)
        if 'category' in full_df.columns and 'category' not in cat_cols:
            cat_cols.append('category')
    
    logger.info(f"Using datetime column: {datetime_col}")
    logger.info(f"Using target column: {target_col}")
    logger.info(f"Using text columns: {text_cols}")
    logger.info(f"Using categorical columns: {cat_cols}")

    # Split data
    train_df, eval_df, test_df = time_based_split(
        full_df, datetime_col, train_ratio, eval_ratio, test_ratio
    )

    # Extract targets
    y_train = train_df[target_col].values
    y_eval = eval_df[target_col].values
    y_test = test_df[target_col].values

    # Prepare features (drop target and any landing page columns)
    drop_cols = [col for col in [target_col, 'Landing Page', 'landing_page'] if col in full_df.columns]
    
    train_df = train_df.drop(columns=drop_cols)
    eval_df = eval_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    # Clean text (only if text columns exist in the data)
    existing_text_cols = [col for col in text_cols if col in train_df.columns]
    if existing_text_cols:
        cleaner = TextCleaner(text_columns=existing_text_cols)
        train_df = cleaner.fit_transform(train_df)
        eval_df = cleaner.transform(eval_df)
        test_df = cleaner.transform(test_df)

        # Extract text stats
        stats_extractor = TextStatsExtractor(text_columns=existing_text_cols)
        train_stats = stats_extractor.fit_transform(train_df)
        eval_stats = stats_extractor.transform(eval_df)
        test_stats = stats_extractor.transform(test_df)
    else:
        logger.warning("No text columns found for processing")
        train_stats = pd.DataFrame()
        eval_stats = pd.DataFrame()
        test_stats = pd.DataFrame()
        cleaner = None
        stats_extractor = None

    # DateTime Features
    dt_extractor = DateTimeFeatures(date_column=datetime_col)
    train_dt = dt_extractor.fit_transform(train_df)
    eval_dt = dt_extractor.transform(eval_df)
    test_dt = dt_extractor.transform(test_df)

    # Initialize feature matrices
    feature_matrices_train = []
    feature_matrices_eval = []
    feature_matrices_test = []
    
    # TF-IDF (if we have text columns)
    if existing_text_cols and 'title' in existing_text_cols:
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        tfidf_title_train = tfidf_vectorizer.fit_transform(train_df['title'])
        tfidf_title_eval = tfidf_vectorizer.transform(eval_df['title'])
        tfidf_title_test = tfidf_vectorizer.transform(test_df['title'])
        
        feature_matrices_train.append(tfidf_title_train)
        feature_matrices_eval.append(tfidf_title_eval)
        feature_matrices_test.append(tfidf_title_test)
    else:
        tfidf_vectorizer = None

    # Category encoding (only if we have categorical columns)
    existing_cat_cols = [col for col in cat_cols if col in train_df.columns]
    if existing_cat_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        cat_train = ohe.fit_transform(train_df[existing_cat_cols])
        cat_eval = ohe.transform(eval_df[existing_cat_cols])
        cat_test = ohe.transform(test_df[existing_cat_cols])
        
        feature_matrices_train.append(cat_train)
        feature_matrices_eval.append(cat_eval)
        feature_matrices_test.append(cat_test)
        
        logger.info(f"Categorical encoding created {cat_train.shape[1]} features from columns: {existing_cat_cols}")
    else:
        ohe = None

    # Add stats features
    if not train_stats.empty:
        feature_matrices_train.append(sparse.csr_matrix(train_stats))
        feature_matrices_eval.append(sparse.csr_matrix(eval_stats))
        feature_matrices_test.append(sparse.csr_matrix(test_stats))

    # Add datetime features
    feature_matrices_train.append(sparse.csr_matrix(train_dt))
    feature_matrices_eval.append(sparse.csr_matrix(eval_dt))
    feature_matrices_test.append(sparse.csr_matrix(test_dt))

    # Embeddings (if we have text columns)
    embedder = None
    if existing_text_cols:
        embedder = EmbeddingVectorizer()
        
        if 'title' in existing_text_cols:
            logger.info("Generating title embeddings...")
            title_embed_train = embedder.fit_transform(train_df['title'].tolist())
            title_embed_eval = embedder.transform(eval_df['title'].tolist())
            title_embed_test = embedder.transform(test_df['title'].tolist())
            
            feature_matrices_train.append(sparse.csr_matrix(title_embed_train))
            feature_matrices_eval.append(sparse.csr_matrix(title_embed_eval))
            feature_matrices_test.append(sparse.csr_matrix(title_embed_test))
        
        if 'title' in existing_text_cols and 'subtitle' in existing_text_cols:
            logger.info("Generating combined title+subtitle embeddings...")
            combined_text_train = (train_df['title'] + " " + train_df['subtitle']).tolist()
            combined_text_eval = (eval_df['title'] + " " + eval_df['subtitle']).tolist()
            combined_text_test = (test_df['title'] + " " + test_df['subtitle']).tolist()
            
            combined_embed_train = embedder.transform(combined_text_train)
            combined_embed_eval = embedder.transform(combined_text_eval)
            combined_embed_test = embedder.transform(combined_text_test)
            
            feature_matrices_train.append(sparse.csr_matrix(combined_embed_train))
            feature_matrices_eval.append(sparse.csr_matrix(combined_embed_eval))
            feature_matrices_test.append(sparse.csr_matrix(combined_embed_test))

    # Stack all features
    if feature_matrices_train:
        X_train = sparse.hstack(feature_matrices_train).tocsr()
        X_eval = sparse.hstack(feature_matrices_eval).tocsr()
        X_test = sparse.hstack(feature_matrices_test).tocsr()
        
        logger.info(f"Final feature matrix shape - Train: {X_train.shape}, Eval: {X_eval.shape}, Test: {X_test.shape}")
    else:
        # Fallback: just use datetime features
        X_train = sparse.csr_matrix(train_dt)
        X_eval = sparse.csr_matrix(eval_dt)
        X_test = sparse.csr_matrix(test_dt)
        
        logger.warning("Using only datetime features - consider adding more feature types")

    # Preprocessor object
    preprocessor = {
        'cleaner': cleaner,
        'stats_extractor': stats_extractor,
        'dt_extractor': dt_extractor,
        'tfidf_vectorizer': tfidf_vectorizer,
        'onehot_encoder': ohe,
        'embedder': embedder,
        'datetime_col': datetime_col,
        'target_col': target_col,
        'text_cols': existing_text_cols,
        'cat_cols': existing_cat_cols
    }

    # Save processed data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    save_processed_data(X_train, X_test, X_eval, y_train, y_test, y_eval, preprocessor, output_dir)
    return X_train, X_test, X_eval, y_train, y_test, y_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess CSV data for SVR training')
    parser.add_argument('input', type=str, help='Path to input CSV file or directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='processed_data', 
                        help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--eval_ratio', type=float, default=0.15, 
                        help='Evaluation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, 
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--datetime_col', type=str, default=None,
                        help='Name of the datetime column (auto-detected if not provided)')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Name of the target column (auto-detected if not provided)')
    parser.add_argument('--text_cols', type=str, nargs='*', default=None,
                        help='Names of text columns (auto-detected if not provided)')
    parser.add_argument('--cat_cols', type=str, nargs='*', default=None,
                        help='Names of categorical columns (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting preprocessing...")
        preprocess_data(
            args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            datetime_col=args.datetime_col,
            target_col=args.target_col,
            text_cols=args.text_cols,
            cat_cols=args.cat_cols
        )
        logger.info("✅ Preprocessing completed successfully")
    except Exception as e:
        logger.exception(f"Preprocessing failed: {str(e)}")
        sys.exit(1)