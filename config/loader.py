"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from ..utils.logger import get_logger

logger = get_logger()


class ConfigLoader:
    """Configuration loader for loading YAML and JSON config files."""
    
    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dict containing the configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file is not valid YAML
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        logger.debug(f"Loading YAML config from {file_path}")
        with open(file_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML config: {e}")
                raise
                
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dict containing the configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        logger.debug(f"Loading JSON config from {file_path}")
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON config: {e}")
                raise
                
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from either YAML or JSON file based on extension.
        
        Args:
            file_path: Path to the config file
            
        Returns:
            Dict containing the configuration
            
        Raises:
            ValueError: If the file extension is not supported
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in [".yaml", ".yml"]:
            return cls.load_yaml(file_path)
        elif file_path.suffix.lower() == ".json":
            return cls.load_json(file_path)
        else:
            raise ValueError(
                f"Unsupported config file extension: {file_path.suffix}. "
                "Supported extensions: .yaml, .yml, .json"
            )
