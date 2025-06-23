"""Example script demonstrating regression with the ML Library."""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_library.models import LinearRegression
from ml_library.training import Trainer
from ml_library.evaluation import RegressionMetrics
from ml_library.inference import Predictor
from ml_library.utils.logger import setup_logger

# Set up logging
setup_logger(level="INFO")

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
trainer = Trainer(model, validation_split=0.0)  # We already have a test set
trainer.train(X_train_scaled, y_train)

# Make predictions
predictor = Predictor(model)
y_pred = predictor.predict(X_test_scaled)

# Evaluate model
metrics = RegressionMetrics()
results = metrics.evaluate(y_test, y_pred)

print("\n----- Evaluation Results -----")
print(f"Mean Absolute Error (MAE): {results['mae']:.4f}")
print(f"Mean Squared Error (MSE): {results['mse']:.4f}")
print(f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
print(f"R² Score: {results['r2']:.4f}")
print(f"Explained Variance: {results['explained_variance']:.4f}")
