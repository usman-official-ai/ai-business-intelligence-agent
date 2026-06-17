"""Model training module for industry classification."""

import logging
import json
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training and evaluation of classification models."""
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.models = {
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=2000
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            ),
            'MultinomialNB': MultinomialNB(alpha=1.0)
        }
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        self.metrics = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            if model_name == 'MultinomialNB':
                X_train_nb = np.clip(X_train, 0, None)
                X_test_nb = np.clip(X_test, 0, None)
                model.fit(X_train_nb, y_train)
                y_pred = model.predict(X_test_nb)
                cv_scores = cross_val_score(model, X_train_nb, y_train, cv=5)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'y_true': y_test,
                'y_pred': y_pred
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[self.best_model_name]['model']
        self.metrics = results
        
        logger.info(f"Best model: {self.best_model_name}")
        return results
    
    def save_models(self, model_dir: str = "models") -> None:
        """Save all trained models and metadata."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, data in self.metrics.items():
            model_path = Path(model_dir) / f"{model_name}.joblib"
            joblib.dump(data['model'], model_path)
        
        best_model_path = Path(model_dir) / "best_model.joblib"
        joblib.dump(self.best_model, best_model_path)
        
        metrics_path = Path(model_dir) / "metrics.json"
        metrics_to_save = {
            name: {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in data.items()
                if k not in ['model', 'y_true', 'y_pred']
            }
            for name, data in self.metrics.items()
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        classes_path = Path(model_dir) / "classes.json"
        with open(classes_path, 'w') as f:
            json.dump(list(self.best_model.classes_), f)
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_model(self, model_path: str) -> Any:
        """Load a trained model from disk."""
        return joblib.load(model_path)
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix for a specific model."""
        if model_name not in self.metrics:
            raise ValueError(f"Model {model_name} not found")
        
        data = self.metrics[model_name]
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.best_model.classes_,
                    yticklabels=self.best_model.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_performance_report(self) -> str:
        """Generate performance report."""
        report = "=" * 80 + "\n"
        report += "MODEL PERFORMANCE REPORT\n"
        report += "=" * 80 + "\n\n"
        report += f"Best Model: {self.best_model_name}\n"
        report += "-" * 40 + "\n"
        
        for model_name, metrics in self.metrics.items():
            report += f"\n{model_name}:\n"
            report += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
            report += f"  Precision: {metrics['precision']:.4f}\n"
            report += f"  Recall:    {metrics['recall']:.4f}\n"
            report += f"  F1 Score:  {metrics['f1_score']:.4f}\n"
            report += f"  CV Mean:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n"
        
        return report