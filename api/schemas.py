"""Pydantic schemas for API request/response validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class CompanyFeatures(BaseModel):
    """Company features for prediction."""
    website_exists: int = Field(..., ge=0, le=1, description="Website exists (0 or 1)")
    contact_form: int = Field(..., ge=0, le=1, description="Contact form exists (0 or 1)")
    services_count: int = Field(..., ge=1, le=20, description="Number of services (1-20)")
    country_score: int = Field(..., ge=1, le=3, description="Country score (1-3)")
    
    @validator('website_exists', 'contact_form')
    def validate_binary(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError('Value must be 0 or 1')
        return v


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    company_name: str = Field(..., description="Company name")
    description: str = Field(..., description="Company description text")
    features: CompanyFeatures = Field(..., description="Company features")


class TopPrediction(BaseModel):
    """Top prediction result."""
    industry: str
    probability: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    company_name: str
    predicted_industry: str
    confidence: float
    top_predictions: List[TopPrediction]
    lead_score: Dict[str, Any]
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    results: List[Dict[str, Any]]