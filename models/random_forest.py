"""Random Forest model implementation."""

from typing import Dict, List, Optional, Union, Any

import numpy as np

from utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class RandomForestRegressor(BaseModel):
    """Random Forest implementation for regression.
    
    This model builds an ensemble of decision trees and averages their predictions.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the Random Forest regressor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            max_features: Number of features to consider for the best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            **kwargs
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimator_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.ensemble import RandomForestRegressor as SklearnRF
            self.SklearnRF = SklearnRF
        except ImportError:
            logger.error("scikit-learn is required for RandomForestRegressor")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Fit the random forest model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting RandomForestRegressor with {self.n_estimators} estimators, "
            f"max_depth={self.max_depth}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.estimator_.fit(X, y)
        self._fitted = True
        
        # Log feature importances
        if hasattr(self.estimator_, 'feature_importances_'):
            importances = self.estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            top_k = min(10, len(indices))
            logger.info(f"Top {top_k} feature importances:")
            for i in range(top_k):
                logger.info(f"Feature {indices[i]}: {importances[indices[i]]:.6f}")
        
        logger.info("RandomForestRegressor fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the random forest model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted values
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        return self.estimator_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X: Test samples
            y: True values
            
        Returns:
            R^2 score
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        y = np.array(y)
        
        return self.estimator_.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }
    
    def _validate_data(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Validate the input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
            
        return X


class RandomForestClassifier(BaseModel):
    """Random Forest implementation for classification.
    
    This model builds an ensemble of decision trees and uses majority voting for classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            max_features: Number of features to consider for the best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            **kwargs
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimator_ = None
        self.classes_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.ensemble import RandomForestClassifier as SklearnRF
            self.SklearnRF = SklearnRF
        except ImportError:
            logger.error("scikit-learn is required for RandomForestClassifier")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Fit the random forest model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting RandomForestClassifier with {self.n_estimators} estimators, "
            f"max_depth={self.max_depth}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.SklearnRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        self._fitted = True
        
        # Log feature importances
        if hasattr(self.estimator_, 'feature_importances_'):
            importances = self.estimator_.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            top_k = min(10, len(indices))
            logger.info(f"Top {top_k} feature importances:")
            for i in range(top_k):
                logger.info(f"Feature {indices[i]}: {importances[indices[i]]:.6f}")
        
        logger.info("RandomForestClassifier fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        return self.estimator_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Probability of each class for each sample
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        return self.estimator_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy on the given test data and labels.
        
        Args:
            X: Test samples
            y: True labels
            
        Returns:
            Accuracy score
        """
        if not self._fitted or self.estimator_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self._validate_data(X)
        y = np.array(y)
        
        return self.estimator_.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dict of model parameters
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }
    
    def _validate_data(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """Validate the input data.
        
        Args:
            X: Input data
            
        Returns:
            Validated numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
            
        return X
