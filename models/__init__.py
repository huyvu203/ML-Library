"""Model implementations."""

from .base import BaseModel
from .regression import LinearRegression
from .classification import LogisticRegression
from .random_forest import RandomForestRegressor, RandomForestClassifier
from .tree_models import DecisionTreeRegressor, DecisionTreeClassifier
from .xgboost_models import XGBoostRegressor, XGBoostClassifier
from .svm_models import SVMRegressor, SVMClassifier
from .knn_models import KNNRegressor, KNNClassifier

__all__ = [
    "BaseModel",
    # Linear models
    "LinearRegression",
    "LogisticRegression",
    # Tree models
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    # Ensemble models
    "RandomForestRegressor",
    "RandomForestClassifier",
    "XGBoostRegressor",
    "XGBoostClassifier",
    # SVM models
    "SVMRegressor",
    "SVMClassifier",
    # KNN models
    "KNNRegressor",
    "KNNClassifier"
]
