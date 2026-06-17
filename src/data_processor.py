"""Data processing module for cleaning and preprocessing company descriptions."""

import re
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class DataProcessor:
    """Handles data cleaning and preprocessing for company descriptions."""
    
    def __init__(self) -> None:
        """Initialize the DataProcessor with stopwords and stemmer."""
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.industries = [
            'AI', 'SaaS', 'Healthcare', 'Fintech', 'CleanTech',
            'EdTech', 'RetailTech', 'Logistics', 'HRTech', 'Cybersecurity',
            'PropTech', 'Automotive', 'FoodTech', 'Energy', 'Telecommunications',
            'Government', 'Aerospace', 'Biotech', 'NanoTech', 'Tourism',
            'Manufacturing'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def tokenize_and_stem(self, text: str) -> str:
        """Tokenize and stem text."""
        try:
            tokens = word_tokenize(text)
            stemmed_tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]
            return ' '.join(stemmed_tokens)
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            return ""
    
    def preprocess_description(self, description: str) -> str:
        """Complete preprocessing pipeline for company descriptions."""
        if not description or not isinstance(description, str):
            logger.warning(f"Invalid description: {description}")
            return ""
        try:
            cleaned = self.clean_text(description)
            processed = self.tokenize_and_stem(cleaned)
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing description: {str(e)}")
            return ""
    
    def create_synthetic_data(self, n_samples: int = 150) -> pd.DataFrame:
        """Create synthetic dataset for development and testing."""
        industry_keywords = {
            'AI': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'ai'],
            'SaaS': ['software as a service', 'cloud computing', 'subscription', 'platform', 'saas'],
            'Healthcare': ['medical', 'health', 'patient', 'clinical', 'hospital', 'healthcare'],
            'Fintech': ['financial', 'banking', 'payment', 'finance', 'cryptocurrency', 'fintech'],
            'CleanTech': ['sustainable', 'renewable', 'green', 'clean', 'environmental', 'solar'],
            'EdTech': ['education', 'learning', 'school', 'student', 'teacher', 'edtech'],
            'RetailTech': ['retail', 'ecommerce', 'shopping', 'store', 'commerce', 'retailtech'],
            'Logistics': ['supply chain', 'delivery', 'shipping', 'warehouse', 'logistics'],
            'HRTech': ['human resources', 'recruitment', 'talent', 'hiring', 'hr'],
            'Cybersecurity': ['security', 'cyber', 'encryption', 'network security', 'firewall'],
            'PropTech': ['real estate', 'property', 'housing', 'mortgage', 'proptech'],
            'Automotive': ['automotive', 'car', 'vehicle', 'auto', 'transportation'],
            'FoodTech': ['food', 'restaurant', 'delivery', 'cuisine', 'foodtech'],
            'Energy': ['energy', 'electricity', 'power', 'renewable', 'grid'],
            'Telecommunications': ['telecom', 'communication', 'network', 'mobile', 'internet'],
            'Government': ['government', 'public sector', 'civic', 'regulatory', 'municipal'],
            'Aerospace': ['aerospace', 'aviation', 'space', 'satellite', 'aircraft'],
            'Biotech': ['biotech', 'genetics', 'biology', 'pharmaceutical', 'dna'],
            'NanoTech': ['nanotech', 'nanoscale', 'molecular', 'nanomaterials', 'quantum'],
            'Tourism': ['travel', 'tourism', 'hotel', 'tour', 'destination'],
            'Manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'materials']
        }
        
        data = []
        companies = [
            "Innovate", "Tech", "Solutions", "Global", "Digital", 
            "Smart", "Future", "Next", "Vanguard", "Synergy",
            "Prime", "Elite", "Pioneer", "Vertex", "Apex", "Zenith"
        ]
        
        np.random.seed(42)
        
        for i in range(n_samples):
            industry = np.random.choice(self.industries)
            
            # Generate company name
            company_name = f"{np.random.choice(companies)} {np.random.choice(['Corp', 'Inc', 'LLC', 'Group', 'Holdings'])}"
            
            # Generate description
            keywords = industry_keywords[industry]
            selected_keywords = np.random.choice(keywords, np.random.randint(2, 4), replace=False)
            description = (
                f"{company_name} is a leading company specializing in "
                f"{', '.join(selected_keywords)}. We provide innovative "
                f"solutions to help businesses grow and succeed."
            )
            
            # Generate features
            website_exists = np.random.choice([0, 1], p=[0.1, 0.9])
            contact_form = np.random.choice([0, 1], p=[0.2, 0.8]) if website_exists else 0
            services_count = np.random.randint(1, 21)
            country_score = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            
            data.append({
                'company_name': company_name,
                'description': description,
                'industry': industry,
                'website_exists': website_exists,
                'contact_form': contact_form,
                'services_count': services_count,
                'country_score': country_score
            })
        
        df = pd.DataFrame(data)
        
        # Ensure balanced distribution
        for industry in self.industries:
            if len(df[df['industry'] == industry]) < 3:
                additional = 5 - len(df[df['industry'] == industry])
                for _ in range(additional):
                    company_name = f"{np.random.choice(companies)} {np.random.choice(['Corp', 'Inc'])}"
                    keywords = industry_keywords[industry]
                    selected_keywords = np.random.choice(keywords, 3, replace=False)
                    description = (
                        f"{company_name} is a leading company specializing in "
                        f"{', '.join(selected_keywords)}."
                    )
                    df.loc[len(df)] = {
                        'company_name': company_name,
                        'description': description,
                        'industry': industry,
                        'website_exists': np.random.choice([0, 1], p=[0.1, 0.9]),
                        'contact_form': np.random.choice([0, 1], p=[0.2, 0.8]),
                        'services_count': np.random.randint(1, 21),
                        'country_score': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                    }
        
        return df