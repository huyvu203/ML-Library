"""Example script demonstrating classification with the ML Library."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_library.models import LogisticRegression
from ml_library.training import Trainer
from ml_library.evaluation import ClassificationMetrics
from ml_library.inference import Predictor
from ml_library.utils.logger import setup_logger

# Set up logging
setup_logger(level="INFO")

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000, 
    n_features=10,
    n_classes=2, 
    n_informative=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LogisticRegression(
    learning_rate=0.05, 
    n_iterations=1000,
    penalty="l2",
    C=1.0
)
trainer = Trainer(model, validation_split=0.0)  # We already have a test set
trainer.train(X_train_scaled, y_train)

# Make predictions
predictor = Predictor(model)
results = predictor.predict(X_test_scaled, return_probabilities=True)
y_pred = results["predictions"]
y_prob = results["probabilities"]

# Evaluate model
metrics = ClassificationMetrics()
eval_results = metrics.evaluate(y_test, y_pred, y_prob)

print("\n----- Evaluation Results -----")
print(f"Accuracy: {eval_results['accuracy']:.4f}")
print(f"Precision: {eval_results['precision']:.4f}")
print(f"Recall: {eval_results['recall']:.4f}")
print(f"F1 Score: {eval_results['f1']:.4f}")
print(f"ROC AUC: {eval_results.get('roc_auc', 'N/A')}")
print("\nConfusion Matrix:")
cm = eval_results["confusion_matrix"]
print(f"[[{cm[0][0]}, {cm[0][1]}]")
print(f" [{cm[1][0]}, {cm[1][1]}]]")
