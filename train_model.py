"""
Model Training Script - Fixed Feature Consistency
"""

from src.data_processor import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer
import pandas as pd
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("🤖 AI Business Intelligence Agent - Model Training")
print("=" * 60)

# Step 1: Create synthetic data
print("\n📊 Creating synthetic data...")
dp = DataProcessor()
df = dp.create_synthetic_data(300)
print(f"✅ Created {len(df)} company records")

# Step 2: Save data
print("\n💾 Saving data...")
df.to_csv("data/raw/companies.csv", index=False)
print("✅ Data saved to data/raw/companies.csv")

# Step 3: Process data
print("\n🔧 Processing text data...")
texts = df["description"].apply(dp.preprocess_description)
features = df[["website_exists", "contact_form", "services_count", "country_score"]]
labels = df["industry"]
print("✅ Text processing complete")

# Step 4: Extract features with FIXED parameters
print("\n🔬 Extracting features with TF-IDF...")
fe = FeatureExtractor(
    max_features=5000,
    ngram_range=(1, 2)
)
X = fe.fit_transform(texts, features)
fe.save("models/feature_extractor.joblib")
print(f"✅ Feature extractor saved with {X.shape[1]} features")

# Step 5: Train models
print("\n🤖 Training models...")
trainer = ModelTrainer()
results = trainer.train_and_evaluate(X, labels, test_size=0.2)
trainer.save_models()
print("✅ Models saved to models/ folder")

# Step 6: Display results
print("\n" + "=" * 60)
print("📊 Performance Summary")
print("=" * 60)

for model_name in results:
    metrics = results[model_name]
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  CV Mean:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")

print("\n" + "=" * 60)
print("✅ Best Model:", trainer.best_model_name)
print(f"✅ Features shape: {X.shape}")
print("=" * 60)
print("\n🎉 Model training completed successfully!")