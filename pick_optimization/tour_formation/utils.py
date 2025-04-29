"""
Utility functions for tour formation.

This module provides utility functions for configuration management,
environment variable handling, and logging setup.
"""

# Standard library imports
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

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
            required_sections = ['global', 'slack_calculation', 'feature_engineering', 'clustering', 'tour_formation']
            missing_sections = [section for section in required_sections if section not in config]
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {str(e)}")
    
    @staticmethod
    def setup_logging(config: Dict[str, Any], output_dir: Optional[str] = None) -> logging.Logger:
        """
        Setup logging configuration with both console and file handlers.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing logging settings
        output_dir : Optional[str], default=None
            Directory where log files will be written
            
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        # Get log level from config, default to INFO if not specified
        log_level = config.get('logging', {}).get('level', 'INFO')
        
        # Convert string log level to numeric level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)     
        root_logger.handlers = [] 
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            config['logging']['format']
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if output_dir is specified
        if output_dir:
            # Create logs directory
            log_dir = output_dir
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file path with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            log_file = os.path.join(log_dir, f'tour_formation_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            root_logger.debug(f"Log file created at: {log_file}")
        
        return root_logger

def load_model_config(input_dir:str) -> Dict[str, Any]:
    """Load configuration for the current model."""
    
    config_path = os.path.join(input_dir, 'tour_formation_config.yaml')
    config = ConfigManager.load_config(config_path)
    
    return config 