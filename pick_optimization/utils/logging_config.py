import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class LoggerFactory:
    """Factory for creating standardized loggers across the application."""
    
    _initialized = False
    _config = None
    _workflow_timestamps: Dict[str, str] = {}  # Store timestamps per workflow
    
    @classmethod
    def initialize(cls, config: Dict[str, Any], workflow_name: Optional[str] = None) -> None:
        """Initialize the logging system with configuration."""
        if cls._initialized:
            return
            
        cls._config = config
        logging_config = config.get('logging', {})
        
        # Create logs directory structure under data folder
        log_base_dir = Path(logging_config.get('log_base_dir', 'logs'))
        log_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create workflow-specific directories
        for workflow in ['main', 'tour_formation', 'tour_allocation']:
            (log_base_dir / workflow).mkdir(parents=True, exist_ok=True)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = logging_config.get('console_level', 'INFO').upper()
        console_handler.setLevel(getattr(logging, console_level))
        
        console_format = logging_config.get('console_format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file handler
        main_log_file = log_base_dir / 'main' / 'main.log'
        file_handler = logging.FileHandler(main_log_file)
        file_level = logging_config.get('file_level', 'INFO').upper()
        file_handler.setLevel(getattr(logging, file_level))
        
        file_format = logging_config.get('file_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        cls._initialized = True
        
        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info(f"Logging system initialized. Main log: {main_log_file}")
        if workflow_name:
            logger.info(f"Workflow: {workflow_name}")
    
    @classmethod
    def get_logger(cls, name: str, workflow: Optional[str] = None) -> logging.Logger:
        """Get a logger with optional workflow-specific file handler."""
        if not cls._initialized:
            # Initialize with default config if not already done
            default_config = {
                'logging': {
                    'log_base_dir': 'data/logs',
                    'console_level': 'INFO',
                    'file_level': 'DEBUG'
                }
            }
            cls.initialize(default_config)
        
        logger = logging.getLogger(name)
        
        # Add workflow-specific file handler if specified
        if workflow and cls._config:
            cls._add_workflow_handler(logger, workflow)
        
        return logger
    
    @classmethod
    def _add_workflow_handler(cls, logger: logging.Logger, workflow: str) -> None:
        """Add workflow-specific file handler to logger."""
        # Check if workflow handler already exists
        handler_name = f"{workflow}_handler"
        for handler in logger.handlers:
            if hasattr(handler, 'name') and handler.name == handler_name:
                return
        
        logging_config = cls._config.get('logging', {})
        log_base_dir = Path(logging_config.get('log_base_dir', 'data/logs'))
        
        # Get or create timestamp for this workflow
        if workflow not in cls._workflow_timestamps:
            cls._workflow_timestamps[workflow] = datetime.now().strftime('%Y%m%d_%H%M')
        
        timestamp = cls._workflow_timestamps[workflow]
        workflow_log_file = log_base_dir / workflow / f"{workflow}_{timestamp}.log"
        
        # Ensure the workflow directory exists
        workflow_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        workflow_handler = logging.FileHandler(workflow_log_file)
        workflow_handler.name = handler_name
        workflow_handler.setLevel(logging.DEBUG)
        
        file_format = logging_config.get('file_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        workflow_formatter = logging.Formatter(file_format)
        workflow_handler.setFormatter(workflow_formatter)
        
        logger.addHandler(workflow_handler)
        logger.info(f"Added workflow log file: {workflow_log_file}")


def setup_logging(config: Dict[str, Any], workflow_name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration - main entry point for logging initialization.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing logging settings
    workflow_name : Optional[str]
        Name of the workflow (tour_formation, tour_allocation, etc.)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    LoggerFactory.initialize(config, workflow_name)
    return LoggerFactory.get_logger(__name__, workflow_name)


def get_logger(name: str, workflow: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with optional workflow-specific logging.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    workflow : Optional[str]
        Workflow name for workflow-specific logging
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return LoggerFactory.get_logger(name, workflow)


# Convenience function for common usage patterns
def get_workflow_logger(name: str, workflow: str) -> logging.Logger:
    """Get a logger with workflow-specific file logging enabled."""
    return LoggerFactory.get_logger(name, workflow)
