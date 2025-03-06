import logging
import os
from typing import Dict, Any

def get_log_level(level: str) -> int:
    return getattr(logging, level.upper())

def setup_logging(config: Dict[str, Any], workflow_name: str = None) -> logging.Logger:
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config['logging']['main_log_file'])
    os.makedirs(log_dir, exist_ok=True)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    root_logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(get_log_level(config['logging']['console_level']))
    console_formatter = logging.Formatter(config['logging']['log_format'])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main file handler
    file_handler = logging.FileHandler(config['logging']['main_log_file'])
    file_handler.setLevel(get_log_level(config['logging']['file_level']))
    file_formatter = logging.Formatter(config['logging']['log_format'])
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # If it's a specific workflow and workflow logs are enabled
    if workflow_name and config['logging']['enable_workflow_logs']:
        workflow_log_dir = config['logging']['workflow_log_dir']
        os.makedirs(workflow_log_dir, exist_ok=True)
        workflow_log_file = os.path.join(workflow_log_dir, f"{workflow_name}.log")
        workflow_handler = logging.FileHandler(workflow_log_file)
        workflow_handler.setLevel(logging.DEBUG)
        workflow_formatter = logging.Formatter(config['logging']['log_format'])
        workflow_handler.setFormatter(workflow_formatter)
        
        # Create a logger specific to this workflow
        workflow_logger = logging.getLogger(workflow_name)
        workflow_logger.setLevel(logging.DEBUG)
        workflow_logger.handlers = []  # Clear any existing handlers
        workflow_logger.addHandler(workflow_handler)
        workflow_logger.propagate = False  # Prevent propagation to root logger
        return workflow_logger

    return root_logger

# Usage example:
# logger = setup_logging(config)  # For main script
# logger = setup_logging(config, 'active_fwd_pick_skus')  # For a specific workflow