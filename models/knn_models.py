"""K-Nearest Neighbors model implementations."""

from typing import Dict, List, Optional, Union, Any

import numpy as np

from utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class KNNRegressor(BaseModel):
    """K-Nearest Neighbors Regressor.
    
    This model implements regression based on k-nearest neighbors.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the KNN regressor.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction
            algorithm: Algorithm used to compute the nearest neighbors
            leaf_size: Leaf size for BallTree or KDTree
            p: Power parameter for the Minkowski metric
            metric: Distance metric to use
            n_jobs: Number of parallel jobs for neighbor search
            **kwargs: Additional parameters
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.n_jobs = n_jobs
        self.estimator_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.neighbors import KNeighborsRegressor
            self.KNeighborsRegressor = KNeighborsRegressor
        except ImportError:
            logger.error("scikit-learn is required for KNNRegressor")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """Fit the KNN regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting KNNRegressor with n_neighbors={self.n_neighbors}, "
            f"weights={self.weights}, algorithm={self.algorithm}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        
        self.estimator_.fit(X, y)
        self._fitted = True
        
        logger.info(
            f"KNNRegressor fitted successfully with {X.shape[0]} training samples"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the KNN regression model.
        
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
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "n_jobs": self.n_jobs,
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


class KNNClassifier(BaseModel):
    """K-Nearest Neighbors Classifier.
    
    This model implements classification based on k-nearest neighbors.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction
            algorithm: Algorithm used to compute the nearest neighbors
            leaf_size: Leaf size for BallTree or KDTree
            p: Power parameter for the Minkowski metric
            metric: Distance metric to use
            n_jobs: Number of parallel jobs for neighbor search
            **kwargs: Additional parameters
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs
        )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.n_jobs = n_jobs
        self.estimator_ = None
        self.classes_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.neighbors import KNeighborsClassifier
            self.KNeighborsClassifier = KNeighborsClassifier
        except ImportError:
            logger.error("scikit-learn is required for KNNClassifier")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Fit the KNN classification model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting KNNClassifier with n_neighbors={self.n_neighbors}, "
            f"weights={self.weights}, algorithm={self.algorithm}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            n_jobs=self.n_jobs
        )
        
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        self._fitted = True
        
        n_classes = len(self.classes_)
        logger.info(
            f"KNNClassifier fitted successfully with {X.shape[0]} training samples "
            f"for {n_classes} classes"
        )
        
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
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "n_jobs": self.n_jobs,
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
