"""
Example demonstrating the use of SVM models for regression and classification tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_library.models import SVMRegressor, SVMClassifier
from ml_library.training import Trainer
from ml_library.evaluation import RegressionMetrics, ClassificationMetrics


def run_svm_regression():
    """Run an example of SVM regression."""
    print("=" * 80)
    print("SVM Regression Example")
    print("=" * 80)
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train different kernel SVM regressors
    kernels = ["linear", "poly", "rbf"]
    for kernel in kernels:
        print(f"\nTraining SVM regressor with {kernel} kernel...")
        model = SVMRegressor(kernel=kernel, C=1.0, epsilon=0.1, random_state=42)
        
        # Train the model
        trainer = Trainer(model, validation_split=0.2)
        trained_model = trainer.train(X_train, y_train)
        
        # Get training metrics
        metrics = trainer.get_metrics()
        print(f"Training score: {metrics['train_score']:.4f}")
        print(f"Validation score: {metrics['val_score']:.4f}")
        
        # Test the model
        y_pred = trained_model.predict(X_test)
        
        # Evaluate the model
        evaluator = RegressionMetrics()
        results = evaluator.evaluate(y_test, y_pred)
        
        print(f"Test R² score: {results['r2']:.4f}")
        print(f"Test MSE: {results['mse']:.4f}")
        print(f"Test RMSE: {results['rmse']:.4f}")
        print(f"Test MAE: {results['mae']:.4f}")


def run_svm_classification():
    """Run an example of SVM classification."""
    print("\n" + "=" * 80)
    print("SVM Classification Example")
    print("=" * 80)
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train different kernel SVM classifiers
    kernels = ["linear", "poly", "rbf"]
    for kernel in kernels:
        print(f"\nTraining SVM classifier with {kernel} kernel...")
        model = SVMClassifier(
            kernel=kernel, 
            C=1.0, 
            probability=True, 
            random_state=42
        )
        
        # Train the model
        trainer = Trainer(model, validation_split=0.2)
        trained_model = trainer.train(X_train, y_train)
        
        # Get training metrics
        metrics = trainer.get_metrics()
        print(f"Training score: {metrics['train_score']:.4f}")
        print(f"Validation score: {metrics['val_score']:.4f}")
        
        # Test the model
        y_pred = trained_model.predict(X_test)
        y_prob = trained_model.predict_proba(X_test)
        
        # Evaluate the model
        evaluator = ClassificationMetrics()
        results = evaluator.evaluate(y_test, y_pred, y_prob)
        
        print(f"Test accuracy: {results['accuracy']:.4f}")
        print(f"Test precision: {results['precision']:.4f}")
        print(f"Test recall: {results['recall']:.4f}")
        print(f"Test F1 score: {results['f1']:.4f}")
        
        # Plot the decision boundary (only for 2D data)
        if X_train.shape[1] == 2 and kernel == "rbf":
            plot_decision_boundary(trained_model, X_test, y_test)


def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of the classifier."""
    # Set up plot
    plt.figure(figsize=(8, 6))
    
    # Create a meshgrid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    # Plot the decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot the testing points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_svm_regression()
    run_svm_classification()
