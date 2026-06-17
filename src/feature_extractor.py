"""Feature extraction module using TF-IDF vectorization."""

import logging
from typing import Tuple, Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Handles feature extraction from text and numerical features."""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Initialize the FeatureExtractor."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english',
            max_df=0.8,
            min_df=2,
            analyzer='word',
            token_pattern=r'[a-zA-Z]+'
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit_transform(self, texts: pd.Series, numerical_features: pd.DataFrame) -> np.ndarray:
        """Fit and transform text and numerical features."""
        text_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        scaled_numerical = self.scaler.fit_transform(numerical_features)
        combined = np.hstack([text_features, scaled_numerical])
        self.is_fitted = True
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return combined
    
    def transform(self, texts: pd.Series, numerical_features: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted transformers."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transforming")
        text_features = self.tfidf_vectorizer.transform(texts).toarray()
        scaled_numerical = self.scaler.transform(numerical_features)
        return np.hstack([text_features, scaled_numerical])
    
    def save(self, path: str) -> None:
        """Save the feature extractor to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Feature extractor saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the feature extractor from disk."""
        data = joblib.load(path)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        self.feature_names = data.get('feature_names', None)
        logger.info(f"Feature extractor loaded from {path}")
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all features."""
        if not self.is_fitted:
            return []
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out().tolist()
        numerical_features = ['website_exists', 'contact_form', 'services_count', 'country_score']
        return tfidf_features + numerical_features
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        if not self.is_fitted:
            return 0
        return len(self.tfidf_vectorizer.get_feature_names_out()) + 4