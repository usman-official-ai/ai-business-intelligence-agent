"""Prediction module for industry classification and lead scoring."""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .feature_extractor import FeatureExtractor
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class LeadPredictor:
    """Handles predictions for industry classification and lead scoring."""
    
    def __init__(self, model_path: str = "models/best_model.joblib"):
        """Initialize the LeadPredictor."""
        self.model_path = Path(model_path)
        self.model = None
        self.feature_extractor = None
        self.data_processor = DataProcessor()
        self.classes = None
        self.is_loaded = False
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the trained model and feature extractor."""
        if model_path:
            self.model_path = Path(model_path)
        
        self.model = joblib.load(self.model_path)
        
        # Load feature extractor
        extractor_path = self.model_path.parent / "feature_extractor.joblib"
        if extractor_path.exists():
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.load(str(extractor_path))
        
        # Load classes
        classes_path = self.model_path.parent / "classes.json"
        if classes_path.exists():
            import json
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        
        self.is_loaded = True
        logger.info("Predictor initialized successfully")
    
    def predict_industry(self, description: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict industry for a company with improved confidence."""
        if not self.is_loaded:
            self.load_model()
        
        # Preprocess description
        processed_text = self.data_processor.preprocess_description(description)
        
        # Prepare features
        text_series = pd.Series([processed_text])
        features_df = pd.DataFrame([[
            features['website_exists'],
            features['contact_form'],
            features['services_count'],
            features['country_score']
        ]], columns=['website_exists', 'contact_form', 'services_count', 'country_score'])
        
        # Extract features
        X = self.feature_extractor.transform(text_series, features_df)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # IMPROVED: Better confidence calculation
        max_prob = float(max(probabilities))
        
        # If confidence is too low, use alternative method
        if max_prob < 0.4:
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-5:][::-1]
            
            # Calculate relative confidence
            sorted_probs = sorted(probabilities, reverse=True)
            if len(sorted_probs) > 1:
                # Confidence based on gap between top and second
                gap = sorted_probs[0] - sorted_probs[1]
                confidence = min(0.95, 0.3 + gap * 2)
            else:
                confidence = max_prob
        else:
            confidence = max_prob
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_predictions = [
            {'industry': self.classes[idx], 'probability': float(probabilities[idx])}
            for idx in top_indices
        ]
        
        return {
            'predicted_industry': prediction,
            'confidence': confidence,
            'top_predictions': top_predictions[:3],  # Top 3 for display
            'all_probabilities': {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            }
        }
    
    def score_lead(self, features: Dict[str, Any], industry: Optional[str] = None) -> Dict[str, Any]:
        """Calculate lead quality score."""
        score = 0.0
        
        if features.get('website_exists', 0) == 1:
            score += 25
        if features.get('contact_form', 0) == 1:
            score += 20
        services = min(features.get('services_count', 1), 20)
        score += min(services * 1.5, 30)
        country_score = features.get('country_score', 1)
        score += country_score * 8
        
        # Industry bonus
        if industry:
            bonuses = {
                'AI': 10, 'Biotech': 10, 'CleanTech': 9, 'Fintech': 9,
                'Cybersecurity': 9, 'Healthcare': 8, 'SaaS': 8, 'Energy': 8,
                'NanoTech': 8, 'EdTech': 7, 'Aerospace': 7, 'PropTech': 6,
                'Automotive': 6, 'Telecommunications': 6, 'Government': 5,
                'Logistics': 5, 'Manufacturing': 5, 'RetailTech': 4,
                'FoodTech': 4, 'HRTech': 4, 'Tourism': 3
            }
            score += bonuses.get(industry, 0)
        
        score = min(score, 100)
        
        if score >= 80:
            category, priority = "High Quality", "Urgent"
        elif score >= 60:
            category, priority = "Medium Quality", "Follow-up"
        elif score >= 40:
            category, priority = "Low Quality", "Nurture"
        else:
            category, priority = "Poor Quality", "Low Priority"
        
        return {
            'score': score,
            'category': category,
            'priority': priority,
            'components': {
                'website_exists': 25 if features.get('website_exists', 0) == 1 else 0,
                'contact_form': 20 if features.get('contact_form', 0) == 1 else 0,
                'services_count': min(features.get('services_count', 1) * 1.5, 30),
                'country_score': features.get('country_score', 1) * 8,
                'industry_bonus': bonuses.get(industry, 0) if industry else 0
            }
        }
    
    def generate_recommendations(self, industry: str, score: float) -> List[str]:
        """Generate sales recommendations."""
        recommendations = []
        
        if score >= 80:
            recommendations.append("🎯 Schedule immediate executive meeting")
            recommendations.append("📊 Prepare customized product demo")
            recommendations.append("👔 Assign senior account executive")
            recommendations.append("📝 Send proposal within 24 hours")
        elif score >= 60:
            recommendations.append("📞 Schedule initial discovery call")
            recommendations.append("📧 Send personalized outreach email")
            recommendations.append("📚 Prepare case studies for relevant industry")
        elif score >= 40:
            recommendations.append("📨 Add to nurture campaign")
            recommendations.append("📰 Send quarterly industry insights")
            recommendations.append("🎓 Invite to relevant webinars")
        else:
            recommendations.append("📨 Add to drip campaign")
            recommendations.append("📅 Follow up in 3-6 months")
        
        # Industry-specific recommendations
        industry_recs = {
            'AI': ["🤖 Highlight machine learning capabilities", "🔒 Discuss data privacy", "🔗 Offer API integration"],
            'SaaS': ["📈 Highlight scalability", "🎁 Offer free trial", "🔗 Discuss integrations"],
            'Healthcare': ["🏥 Emphasize HIPAA compliance", "🔒 Discuss patient data security", "👨‍⚕️ Schedule medical review"],
            'Fintech': ["📋 Discuss regulatory compliance", "🛡️ Highlight fraud detection", "🔒 Offer security audit"],
            'CleanTech': ["🌱 Highlight environmental impact", "💰 Discuss government incentives", "♻️ Offer sustainability consultation"],
            'Cybersecurity': ["🛡️ Emphasize zero-trust architecture", "📋 Offer security assessment", "🔒 Discuss compliance"],
            'EdTech': ["📚 Highlight learning outcomes", "👨‍🏫 Offer teacher training", "📊 Show student engagement"],
            'Logistics': ["📦 Discuss supply chain optimization", "🚚 Offer tracking solutions", "📊 Show efficiency metrics"],
            'HRTech': ["👥 Highlight talent management", "📊 Show retention metrics", "🤝 Offer consulting services"]
        }
        
        for rec in industry_recs.get(industry, []):
            if len(recommendations) < 6:
                recommendations.append(rec)
        
        return recommendations[:6]