"""XGBoost model implementations."""

from typing import Dict, List, Optional, Union, Any

import numpy as np

from ..utils.logger import get_logger
from .base import BaseModel

logger = get_logger()


class XGBoostRegressor(BaseModel):
    """XGBoost implementation for regression.
    
    This model uses gradient boosted trees for regression tasks.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the XGBoost regressor.
        
        Args:
            n_estimators: Number of gradient boosted trees
            learning_rate: Step size shrinkage used in update to prevent overfitting
            max_depth: Maximum depth of a tree
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            **kwargs
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.estimator_ = None
        
        # Import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            logger.error("xgboost is required for XGBoostRegressor")
            raise ImportError(
                "xgboost is not installed. "
                "Install it with 'poetry add xgboost' or 'pip install xgboost'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        """Fit the XGBoost model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting XGBoostRegressor with {self.n_estimators} estimators, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth}"
        )
        
        # Create and fit the XGBoost model
        self.estimator_ = self.xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
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
        
        logger.info("XGBoostRegressor fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the XGBoost model.
        
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
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
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


class XGBoostClassifier(BaseModel):
    """XGBoost implementation for classification.
    
    This model uses gradient boosted trees for classification tasks.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize the XGBoost classifier.
        
        Args:
            n_estimators: Number of gradient boosted trees
            learning_rate: Step size shrinkage used in update to prevent overfitting
            max_depth: Maximum depth of a tree
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            **kwargs
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.estimator_ = None
        self.classes_ = None
        
        # Import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            logger.error("xgboost is required for XGBoostClassifier")
            raise ImportError(
                "xgboost is not installed. "
                "Install it with 'poetry add xgboost' or 'pip install xgboost'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """Fit the XGBoost model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        X = self._validate_data(X)
        y = np.array(y)
        
        logger.debug(
            f"Fitting XGBoostClassifier with {self.n_estimators} estimators, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth}"
        )
        
        # Create and fit the XGBoost model
        self.estimator_ = self.xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
            use_label_encoder=False,  # Avoid deprecation warning
            eval_metric='logloss'  # Default metric for classification
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
        
        logger.info("XGBoostClassifier fitted successfully")
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
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
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
