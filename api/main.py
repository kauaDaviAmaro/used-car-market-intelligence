"""FastAPI application for used car price prediction."""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import numpy as np
import sys
import os

# Handle imports for both module and direct execution
if __name__ == "__main__":
    # When running directly, add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.models import CarFeatures
    from api.model_loader import ModelLoader
    from api.feature_processor import FeatureProcessor
else:
    # When running as module, use relative imports
    from .models import CarFeatures
    from .model_loader import ModelLoader
    from .feature_processor import FeatureProcessor

# Global instances
model_loader = ModelLoader()
feature_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global feature_processor
    model_loader.load()
    feature_processor = FeatureProcessor(model_loader)
    yield
    # Shutdown (if needed in the future)

app = FastAPI(
    title="Used Car Price Predictor API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Used Car Price Predictor API", "version": "1.0.0"}

@app.post("/predict")
async def predict(car: CarFeatures):
    """
    Predict the price of a used car based on its features.
    
    Returns the predicted price in the original scale (not log-transformed).
    """
    model = model_loader.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if feature_processor is None:
        raise HTTPException(status_code=500, detail="Feature processor not initialized")
    
    try:
        # Prepare features
        features_df = feature_processor.prepare_features(car)
        
        # Make prediction (model returns log_price)
        log_price_pred = model.predict(features_df)[0]
        
        # Convert from log scale to actual price
        price_pred = np.expm1(log_price_pred)
        
        return {
            "predicted_price": float(price_pred),
            "predicted_price_log": float(log_price_pred),
            "currency": "BRL"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
