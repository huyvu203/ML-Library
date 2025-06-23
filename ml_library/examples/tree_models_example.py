"""Example script demonstrating tree-based models with the ML Library."""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_library.models import (
    DecisionTreeRegressor,
    RandomForestRegressor,
    XGBoostRegressor,
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBoostClassifier
)
from ml_library.training import Trainer
from ml_library.evaluation import RegressionMetrics, ClassificationMetrics
from ml_library.utils.logger import setup_logger

# Set up logging
setup_logger(level="INFO")

# Demo function for regression models
def test_regression_model(model_class, model_name, **model_params):
    """Test a regression model and print its performance."""
    print(f"\n{'='*20} Testing {model_name} {'='*20}")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = model_class(**model_params)
    trainer = Trainer(model)
    trainer.train(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    metrics = RegressionMetrics()
    results = metrics.evaluate(y_test, y_pred)
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"Explained Variance: {results['explained_variance']:.4f}")


# Demo function for classification models
def test_classification_model(model_class, model_name, **model_params):
    """Test a classification model and print its performance."""
    print(f"\n{'='*20} Testing {model_name} {'='*20}")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = model_class(**model_params)
    trainer = Trainer(model)
    trainer.train(X_train_scaled, y_train)
    
    # Make predictions with probabilities
    y_pred = model.predict(X_test_scaled)
    try:
        y_prob = model.predict_proba(X_test_scaled)
        has_proba = True
    except (AttributeError, NotImplementedError):
        y_prob = None
        has_proba = False
    
    # Evaluate model
    metrics = ClassificationMetrics()
    results = metrics.evaluate(y_test, y_pred, y_prob)
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    # Print confusion matrix
    cm = results["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"[[{cm[0][0]}, {cm[0][1]}]")
    print(f" [{cm[1][0]}, {cm[1][1]}]]")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REGRESSION MODELS COMPARISON")
    print("="*60)
    
    # Test regression models
    test_regression_model(
        DecisionTreeRegressor, 
        "Decision Tree Regressor",
        max_depth=10, 
        random_state=42
    )
    
    test_regression_model(
        RandomForestRegressor, 
        "Random Forest Regressor",
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    
    test_regression_model(
        XGBoostRegressor, 
        "XGBoost Regressor",
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION MODELS COMPARISON")
    print("="*60)
    
    # Test classification models
    test_classification_model(
        DecisionTreeClassifier, 
        "Decision Tree Classifier",
        max_depth=10, 
        random_state=42
    )
    
    test_classification_model(
        RandomForestClassifier, 
        "Random Forest Classifier",
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    
    test_classification_model(
        XGBoostClassifier, 
        "XGBoost Classifier",
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
