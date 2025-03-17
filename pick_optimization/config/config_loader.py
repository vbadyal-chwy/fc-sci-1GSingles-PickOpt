"""
Configuration loader module for pick optimization.

This module provides functions to load and parse configuration files,
including handling of encrypted credentials for DB connections.
"""
import yaml
import os
from typing import Dict, Any, Tuple
from cryptography.fernet import Fernet


def read_secrets(file_path: str) -> Tuple[str, str]:
    """
    Read encrypted credentials from a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing the encryption key and encrypted credentials
        
    Returns
    -------
    Tuple[str, str]
        A tuple containing (key, encrypted_credentials)
    """
    with open(file_path, 'r') as file:
        key = file.readline().strip()
        encrypted_credentials = file.readline().strip()
    return key, encrypted_credentials


def decrypt_credentials(key: str, encrypted_credentials: str) -> str:
    """
    Decrypt credentials using the provided key.
    
    Parameters
    ----------
    key : str
        The encryption key
    encrypted_credentials : str
        The encrypted credentials string
        
    Returns
    -------
    str
        The decrypted credentials
    """
    cipher_suite = Fernet(key)
    decrypted_credentials = cipher_suite.decrypt(encrypted_credentials.encode())
    return decrypted_credentials.decode()


def get_creds(file_path: str) -> Tuple[str, str]:
    """
    Get username and decrypted password from a credentials file.
    
    Parameters
    ----------
    file_path : str
        Path to the credentials file
        
    Returns
    -------
    Tuple[str, str]
        A tuple containing (username, decrypted_password)
    """
    key, encrypted_credentials = read_secrets(file_path)
    decrypted_credentials = decrypt_credentials(key, encrypted_credentials)
    username = os.path.basename(os.path.dirname(file_path))
    return username, decrypted_credentials


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse configuration from YAML files.
    
    This function loads the main configuration file and optionally the Gurobi
    configuration file. It also handles loading and decrypting database credentials.
    
    Parameters
    ----------
    config_path : str
        Path to the main configuration YAML file
        
    Returns
    -------
    Dict[str, Any]
        The parsed configuration dictionary with decrypted credentials
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Replace ${USER} with the actual username in the credentials file paths
    if 'database' in config:
        if 'vertica' in config['database'] and 'credentials_file' in config['database']['vertica']:
            config['database']['vertica']['credentials_file'] = config['database']['vertica']['credentials_file'].replace('${USER}', os.getenv('USER', ''))
        
    # Load and decrypt Vertica credentials
    if 'database' in config and 'vertica' in config['database'] and 'credentials_file' in config['database']['vertica']:
        username, password = get_creds(config['database']['vertica']['credentials_file'])
        config['database']['vertica']['username'] = username
        config['database']['vertica']['password'] = password
    
    
    # Load Gurobi config
    gurobi_config_path = os.path.join(os.path.dirname(config_path), 'gurobi_config.yaml')
    if os.path.exists(gurobi_config_path):
        with open(gurobi_config_path, 'r') as gurobi_config_file:
            config['gurobi'] = yaml.safe_load(gurobi_config_file)
    
    return config 