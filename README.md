# ML Library

A production-ready Python machine learning library with modular components for preprocessing, training, inference, and evaluation.

## Features

- Modular pipeline for data preprocessing, model training, inference, and evaluation
- Support for classification and regression tasks
- Configuration management via YAML and JSON
- Scikit-learn compatible interfaces
- Fully tested with pytest and hypothesis
- Type hinted with mypy compliance
- Structured logging with loguru

## Installation

### Using Poetry (recommended)

```bash
poetry add ml_library
```

### Using pip

```bash
pip install ml_library
```

## Development Setup

This project uses Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/yourusername/ml_library
cd ml_library

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

## Usage Examples

### Basic Regression Example

```python
from ml_library.preprocessing import StandardScaler
from ml_library.models import LinearRegression
from ml_library.training import Trainer
from ml_library.evaluation import RegressionMetrics

# Prepare data
X_train, X_test, y_train, y_test = get_train_test_data()

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LinearRegression()
trainer = Trainer(model)
trainer.train(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate model
metrics = RegressionMetrics()
results = metrics.evaluate(y_test, predictions)
print(f"RMSE: {results['rmse']:.4f}")
print(f"R²: {results['r2']:.4f}")
```

### Loading from Configuration

```python
from ml_library.config import ConfigLoader
from ml_library.models import create_model_from_config

# Load configuration
config = ConfigLoader.load_yaml("config.yaml")

# Create model from config
model = create_model_from_config(config)
```

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=ml_library
```

## Documentation

For full documentation, see [https://ml-library.readthedocs.io](https://ml-library.readthedocs.io).

To build documentation locally:

```bash
cd docs
poetry run sphinx-build -b html . _build/html
```

## License

MIT License
