"""Decision Tree model implementations."""

from typing import Dict, List, Optional, Union, Any

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class DecisionTreeRegressor(BaseModel):
    """Decision Tree implementation for regression.
    
    This model uses a decision tree for regression tasks.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = None,
        criterion: str = "squared_error", 
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the Decision Tree regressor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            max_features: Number of features to consider for the best split
            criterion: The function to measure the quality of a split
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            **kwargs
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.estimator_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.tree import DecisionTreeRegressor as SklearnDT
            self.SklearnDT = SklearnDT
        except ImportError:
            logger.error("scikit-learn is required for DecisionTreeRegressor")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Fit the decision tree model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting DecisionTreeRegressor with max_depth={self.max_depth}, "
            f"criterion={self.criterion}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.SklearnDT(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=self.random_state
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
        
        logger.info("DecisionTreeRegressor fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the decision tree model.
        
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
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "criterion": self.criterion,
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


class DecisionTreeClassifier(BaseModel):
    """Decision Tree implementation for classification.
    
    This model uses a decision tree for classification tasks.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = None,
        criterion: str = "gini",
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the Decision Tree classifier.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at a leaf node
            max_features: Number of features to consider for the best split
            criterion: The function to measure the quality of a split
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            **kwargs
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.estimator_ = None
        self.classes_ = None
        
        # Import scikit-learn implementation
        try:
            from sklearn.tree import DecisionTreeClassifier as SklearnDT
            self.SklearnDT = SklearnDT
        except ImportError:
            logger.error("scikit-learn is required for DecisionTreeClassifier")
            raise ImportError(
                "scikit-learn is not installed. "
                "Install it with 'poetry add scikit-learn' or 'pip install scikit-learn'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit the decision tree model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting DecisionTreeClassifier with max_depth={self.max_depth}, "
            f"criterion={self.criterion}"
        )
        
        # Create and fit the scikit-learn model
        self.estimator_ = self.SklearnDT(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=self.random_state
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
        
        logger.info("DecisionTreeClassifier fitted successfully")
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
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "criterion": self.criterion,
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
