"""FastAPI application for AI Business Intelligence Agent."""

import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.predictor import LeadPredictor
from src.data_processor import DataProcessor
from .schemas import (
    PredictionRequest, PredictionResponse, HealthResponse, 
    TopPrediction, BatchPredictionResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Business Intelligence Agent",
    description="Industry classification and lead scoring API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = LeadPredictor()
data_processor = DataProcessor()

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    try:
        predictor.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Business Intelligence Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=predictor.is_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict industry and score lead."""
    try:
        # Prepare features
        features = {
            'website_exists': request.features.website_exists,
            'contact_form': request.features.contact_form,
            'services_count': request.features.services_count,
            'country_score': request.features.country_score
        }
        
        # Get prediction
        prediction = predictor.predict_industry(request.description, features)
        
        # Score lead
        lead_score = predictor.score_lead(features, prediction['predicted_industry'])
        
        # Generate recommendations
        recommendations = predictor.generate_recommendations(
            prediction['predicted_industry'],
            lead_score['score']
        )
        
        return PredictionResponse(
            company_name=request.company_name,
            predicted_industry=prediction['predicted_industry'],
            confidence=prediction['confidence'],
            top_predictions=[
                TopPrediction(**p) for p in prediction['top_predictions']
            ],
            lead_score=lead_score,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint."""
    try:
        results = []
        for request in requests:
            features = {
                'website_exists': request.features.website_exists,
                'contact_form': request.features.contact_form,
                'services_count': request.features.services_count,
                'country_score': request.features.country_score
            }
            prediction = predictor.predict_industry(request.description, features)
            lead_score = predictor.score_lead(features, prediction['predicted_industry'])
            recommendations = predictor.generate_recommendations(
                prediction['predicted_industry'],
                lead_score['score']
            )
            
            results.append({
                'company_name': request.company_name,
                'predicted_industry': prediction['predicted_industry'],
                'confidence': prediction['confidence'],
                'lead_score': lead_score,
                'recommendations': recommendations
            })
        
        return BatchPredictionResponse(results=results)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)