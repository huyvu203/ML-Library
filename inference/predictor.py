"""Predictor implementation for making model predictions."""

import pickle
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from models.base import BaseModel
from utils.logger import get_logger

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
        
        if return_probabilities and all_probabilities:
            probabilities = np.concatenate(all_probabilities)
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
            
        return predictions
    
    def save_predictions(self, predictions: Union[np.ndarray, Dict[str, np.ndarray]], file_path: Union[str, Path]) -> None:
        """Save predictions to a file.
        
        Args:
            predictions: Predictions to save, either a numpy array or a dict with predictions and probabilities
            file_path: Path where predictions will be saved
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving predictions to {file_path}")
        
        # Check if it's a dictionary with predictions and probabilities
        if isinstance(predictions, dict) and "predictions" in predictions and "probabilities" in predictions:
            # Save as .npz file with multiple arrays
            np.savez(file_path, 
                     predictions=predictions["predictions"], 
                     probabilities=predictions["probabilities"])
        else:
            # For regular arrays, use standard np.save
            np.save(file_path, predictions)
    
    def load_predictions(self, file_path: Union[str, Path]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Load predictions from a file.
        
        Args:
            file_path: Path to the saved predictions
            
        Returns:
            The loaded predictions
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            # Check if .npz extension exists instead
            npz_path = file_path.with_suffix('.npz')
            if npz_path.exists():
                file_path = npz_path
            else:
                raise FileNotFoundError(f"Predictions file not found: {file_path}")
        
        logger.info(f"Loading predictions from {file_path}")
        
        # Handle .npz files (dictionary format)
        if str(file_path).endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            return {
                "predictions": data["predictions"],
                "probabilities": data["probabilities"]
            }
        else:
            # For regular arrays, ensure allow_pickle is True
            return np.load(file_path, allow_pickle=True)
    
    def explain_feature_importance(self, feature_names: Optional[List[str]] = None) -> List[Dict[str, Union[str, float]]]:
        """Get feature importance of the model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            List of dicts with feature names and importance values, sorted by importance
            
        Raises:
            ValueError: If the model doesn't support feature importance
        """
        if not hasattr(self.model, "estimator_") or not hasattr(self.model.estimator_, "feature_importances_"):
            raise ValueError(f"Model {self.model.__class__.__name__} does not support feature importance")
        
        importances = self.model.estimator_.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        if len(feature_names) != len(importances):
            logger.warning("Number of feature names does not match number of importance values")
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create list of dicts with feature names and importances
        feature_importances = [
            {"feature": name, "importance": importance}
            for name, importance in zip(feature_names, importances)
        ]
        
        # Sort by importance (descending)
        feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        
        return feature_importances
