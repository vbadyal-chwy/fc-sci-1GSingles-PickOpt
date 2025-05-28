"""
Utility functions for tour formation.

This module provides utility functions for configuration management,
environment variable handling, and logging setup.
"""

# Standard library imports
from typing import Dict, Any, Union
from pathlib import Path

# Third-party imports
import yaml

class ConfigManager:
    """Manages environment variables and configuration loading for tour formation."""
    
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['global', 'tour_allocation']
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {str(e)}")
    
    # Note: Logging setup has been moved to centralized logging_config.py
    # Use the centralized logging system instead of this method

def load_model_config(input_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration for the current model from the input directory."""
    
    input_dir = normalize_path(input_dir)
    config_path = input_dir / "tour_allocation_config.yaml"
    config = ConfigManager.load_config(config_path)
    
    return config 

def normalize_path(path: Union[str, Path]) -> Path:
    """Converts a string path to a Path object."""
    return Path(path) 