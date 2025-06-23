"""Metrics implementation for model evaluation."""

from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..utils.logger import get_logger

logger = get_logger()


class BaseMetrics:
    """Base class for metrics calculation."""
    
    def __init__(self) -> None:
        """Initialize the metrics calculator."""
        pass
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dict of metric names and values
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class RegressionMetrics(BaseMetrics):
    """Metrics calculator for regression models."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dict of metrics including:
                - mae: Mean Absolute Error
                - mse: Mean Squared Error
                - rmse: Root Mean Squared Error
                - r2: R-squared score
        """
        logger.debug("Calculating regression metrics")
        
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Optional: explained variance score
        explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": explained_variance
        }
        
        logger.info(f"Regression metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        return metrics


class ClassificationMetrics(BaseMetrics):
    """Metrics calculator for classification models."""
    
    def __init__(self, average: str = "binary") -> None:
        """Initialize the classification metrics calculator.
        
        Args:
            average: Averaging method for multiclass problems.
                Options: 'binary' (default), 'micro', 'macro', 'weighted'
        """
        super().__init__()
        self.average = average
        
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Union[np.ndarray, None] = None
    ) -> Dict[str, Union[float, List[List[int]]]]:
        """Evaluate classification model performance.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_prob: Optional predicted probabilities (for ROC-AUC)
            
        Returns:
            Dict of metrics including:
                - accuracy: Accuracy score
                - precision: Precision score
                - recall: Recall score
                - f1: F1 score
                - confusion_matrix: Confusion matrix
                - roc_auc: ROC AUC score (if y_prob provided)
        """
        logger.debug("Calculating classification metrics")
        
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }
        
        # Calculate ROC AUC score if probabilities are provided
        if y_prob is not None:
            try:
                # For binary classification
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                    metrics["roc_auc"] = roc_auc
                # For multiclass
                else:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=self.average)
                    metrics["roc_auc"] = roc_auc
            except (ValueError, IndexError):
                logger.warning("Could not calculate ROC AUC score. Check class labels and probabilities.")
        
        logger.info(
            f"Classification metrics: "
            f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}"
        )
        return metrics
