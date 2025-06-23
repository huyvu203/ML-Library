"""Example script demonstrating configuration loading with the ML Library."""

import os
from pathlib import Path

from ml_library.config import ConfigLoader
from ml_library.models import LinearRegression, LogisticRegression
from ml_library.utils.logger import setup_logger

# Set up logging
setup_logger(level="INFO")

# First, let's create example configuration files
example_dir = Path("examples/configs")
example_dir.mkdir(exist_ok=True, parents=True)

# Create a YAML configuration
yaml_config = """
model:
  type: linear_regression
  params:
    learning_rate: 0.01
    n_iterations: 2000
    fit_intercept: true
    tol: 1.0e-5

training:
  validation_split: 0.2
  random_state: 42

preprocessing:
  scaling: true
"""

with open(example_dir / "regression_config.yaml", "w") as f:
    f.write(yaml_config)

# Create a JSON configuration
json_config = """{
  "model": {
    "type": "logistic_regression",
    "params": {
      "learning_rate": 0.05,
      "n_iterations": 1500,
      "fit_intercept": true,
      "penalty": "l2",
      "C": 0.8
    }
  },
  "training": {
    "validation_split": 0.2,
    "random_state": 42
  },
  "preprocessing": {
    "scaling": true
  }
}"""

with open(example_dir / "classification_config.json", "w") as f:
    f.write(json_config)

# Function to create a model from config
def create_model_from_config(config):
    """Create a model instance from a configuration dictionary.
    
    Args:
        config: A configuration dictionary
        
    Returns:
        An initialized model instance
    """
    model_config = config.get("model", {})
    model_type = model_config.get("type", "")
    model_params = model_config.get("params", {})
    
    if model_type == "linear_regression":
        return LinearRegression(**model_params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Demo loading and using configurations
print("\n----- Loading YAML Configuration -----")
yaml_config_path = example_dir / "regression_config.yaml"
yaml_config = ConfigLoader.load_yaml(yaml_config_path)
print(f"Loaded configuration: {yaml_config}")

# Create model from YAML config
model1 = create_model_from_config(yaml_config)
print(f"\nCreated model: {model1.__class__.__name__}")
print(f"With parameters: {model1.get_params()}")

print("\n----- Loading JSON Configuration -----")
json_config_path = example_dir / "classification_config.json"
json_config = ConfigLoader.load_json(json_config_path)
print(f"Loaded configuration: {json_config}")

# Create model from JSON config
model2 = create_model_from_config(json_config)
print(f"\nCreated model: {model2.__class__.__name__}")
print(f"With parameters: {model2.get_params()}")

# Using the generic loader
print("\n----- Using the generic loader -----")
json_config2 = ConfigLoader.load(json_config_path)
print(f"JSON loaded with generic loader: {json_config2 == json_config}")
