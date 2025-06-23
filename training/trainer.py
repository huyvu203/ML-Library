"""Trainer implementation for model training."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from models.base import BaseModel
from utils.logger import get_logger

logger = get_logger()


class Trainer:
    """Trainer class for training machine learning models.
    
    This class provides functionality to train models,
    including data splitting and logging of training progress.
    """
    
    def __init__(
        self,
        model: BaseModel,
        validation_split: float = 0.2,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the Trainer.
        
        Args:
            model: The model to train
            validation_split: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for training
        """
        self.model = model
        self.validation_split = validation_split
        self.random_state = random_state
        self.training_params = kwargs
        self.metrics: Dict[str, Any] = {}
        
        logger.debug(
            f"Initialized trainer with model {model.__class__.__name__}, "
            f"validation_split={validation_split}, "
            f"random_state={random_state}, "
            f"params={kwargs}"
        )
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> BaseModel:
        """Train the model.
        
        Args:
            X: Training features
            y: Target values
            validation_data: Optional tuple of (X_val, y_val) for validation
            
        Returns:
            The trained model
        """
        logger.info("Starting model training")
        
        # Split data for validation if not provided
        if validation_data is None and self.validation_split > 0:
            X_train, X_val, y_train, y_val = self._train_test_split(
                X, y, test_size=self.validation_split, random_state=self.random_state
            )
            logger.debug(
                f"Split data for validation: "
                f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
                f"X_val: {X_val.shape}, y_val: {y_val.shape}"
            )
        else:
            X_train, y_train = X, y
            if validation_data is not None:
                X_val, y_val = validation_data
                logger.debug(
                    f"Using provided validation data: "
                    f"X_val: {X_val.shape}, y_val: {y_val.shape}"
                )
        
        # Fit the model
        logger.info("Fitting model")
        self.model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_score = self.model.score(X_train, y_train)
        self.metrics["train_score"] = train_score
        logger.info(f"Training score: {train_score:.4f}")
        
        # Evaluate on validation data if available
        if self.validation_split > 0 or validation_data is not None:
            val_score = self.model.score(X_val, y_val)
            self.metrics["val_score"] = val_score
            logger.info(f"Validation score: {val_score:.4f}")
        
        return self.model
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics.
        
        Returns:
            Dict of metrics from the training process
        """
        return self.metrics
        
    def _train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and test/validation sets.
        
        Args:
            X: Features
            y: Target values
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            
        # Get number of samples
        n_samples = len(X)
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        test_samples = int(test_size * n_samples)
        
        # Split indices
        test_idx = indices[:test_samples]
        train_idx = indices[test_samples:]
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        return X_train, X_test, y_train, y_test
