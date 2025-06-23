"""Predictor implementation for making model predictions."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..models.base import BaseModel
from ..utils.logger import get_logger

logger = get_logger()


class Predictor:
    """Predictor class for making predictions with trained models.
    
    This class provides functionality to make predictions with trained models,
    including batch processing and result formatting.
    """
    
    def __init__(self, model: BaseModel) -> None:
        """Initialize the Predictor.
        
        Args:
            model: The trained model to use for predictions
            
        Raises:
            ValueError: If the model is not trained
        """
        self.model = model
        
        # Check if model is trained
        try:
            self.model.check_is_fitted()
        except (ValueError, AttributeError) as e:
            logger.error(f"Model is not properly trained: {e}")
            raise ValueError("Model must be trained before creating a predictor")
            
        logger.debug(f"Initialized predictor with model {model.__class__.__name__}")
        
    def predict(
        self, 
        X: Union[np.ndarray, List[List[float]]],
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions on input data.
        
        Args:
            X: Input features to predict on
            return_probabilities: Whether to return probabilities (for classification)
            
        Returns:
            Predictions as numpy array or dict with predictions and probabilities
        """
        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Log prediction attempt
        logger.debug(f"Making predictions on {len(X)} samples")
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # For classification models, also get probabilities if requested
        if return_probabilities:
            try:
                probabilities = self.model.predict_proba(X)
                return {
                    "predictions": predictions,
                    "probabilities": probabilities
                }
            except (AttributeError, NotImplementedError):
                logger.warning(
                    "Model does not support probability predictions. "
                    "Returning predictions only."
                )
                
        return predictions
        
    def predict_batch(
        self, 
        X: np.ndarray, 
        batch_size: int = 1000,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions in batches to handle large datasets.
        
        Args:
            X: Input features to predict on
            batch_size: Number of samples per batch
            return_probabilities: Whether to return probabilities
            
        Returns:
            Predictions as numpy array or dict with predictions and probabilities
        """
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        # Initialize results containers
        all_predictions = []
        all_probabilities = [] if return_probabilities else None
        
        # Process data in batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            logger.debug(f"Processing batch {i+1}/{n_batches}")
            batch_X = X[start_idx:end_idx]
            
            result = self.predict(batch_X, return_probabilities)
            
            if isinstance(result, dict):
                all_predictions.append(result["predictions"])
                all_probabilities.append(result["probabilities"])
            else:
                all_predictions.append(result)
        
        # Combine batch results
        predictions = np.concatenate(all_predictions)
        
        if all_probabilities:
            probabilities = np.concatenate(all_probabilities)
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
            
        return predictions
