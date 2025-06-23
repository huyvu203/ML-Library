"""Base model implementation."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..utils.logger import get_logger

logger = get_logger()


class BaseModel(ABC):
    """Abstract base class for all models in the library.
    
    This class defines the common interface that all models should implement.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model.
        
        Args:
            **kwargs: Model-specific parameters
        """
        self._fitted = False
        self.params = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with params: {kwargs}")
    
    def check_is_fitted(self) -> None:
        """Check if the model has been fitted.
        
        Raises:
            ValueError: If the model is not fitted.
        """
        if not self._fitted:
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before using this model.")
        return None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values
            
        Returns:
            self: The fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model.
        
        Args:
            X: Data to predict on of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        pass
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Args:
            file_path: Path where the model will be saved
        """
        if not self._fitted:
            logger.warning("Saving an unfitted model")
            
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BaseModel":
        """Load a model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            The loaded model
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        logger.info(f"Loading model from {file_path}")
        with open(file_path, "rb") as f:
            model = pickle.load(f)
            
        if not isinstance(model, cls):
            logger.warning(
                f"Loaded model is of type {type(model).__name__}, "
                f"expected {cls.__name__}"
            )
            
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the model.
        
        Returns:
            Dict of model parameters
        """
        return self.params
