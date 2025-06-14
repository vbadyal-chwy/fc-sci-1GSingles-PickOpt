"""
Data retrieval module for pick optimization. For local runs only!

This module provides functionality to retrieve data from various sources,
including databases and CSV files.
"""
import os
import pandas as pd
import yaml
import vertica_python
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from cryptography.fernet import Fernet


def setup_logging(config: Dict[str, Any] = None, logger_name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary that may contain logging settings
    logger_name : str, default=__name__
        Name of the logger
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_level = logging.INFO
    if config and 'logging' in config and 'level' in config['logging']:
        log_level_str = config['logging']['level'].upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(logger_name)


def load_data_config() -> Dict[str, Any]:
    """Load configuration for the current model from the input directory."""
    
    # Get the parent directory of the current file
    parent_dir = Path(__file__).parent
    
    # Define path to config file in the parent directory
    config_path = parent_dir / "sim_config.yaml"
    
    # Open and safely load the YAML file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    return config


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
    
    if config['global']['input_mode'] == '1':
        # Replace ${USER} with the actual username in the credentials file paths
        if 'database' in config:
            if 'vertica' in config['database'] and 'credentials_file' in config['database']['vertica']:
                config['database']['vertica']['credentials_file'] = config['database']['vertica']['credentials_file'].replace('${USER}', os.getenv('USER', ''))
            
        # Load and decrypt Vertica credentials
        if 'database' in config and 'vertica' in config['database'] and 'credentials_file' in config['database']['vertica']:
            username, password = get_creds(config['database']['vertica']['credentials_file'])
            config['database']['vertica']['username'] = username
            config['database']['vertica']['password'] = password
    return config


class DataPuller:
    """
    Data retrieval class for pick planning optimization.
    
    This class handles retrieving container and slotbook data from
    either a database or CSV files based on the configured input mode.
    """
    
    def __init__(self):
        """
        Initialize the DataPuller with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        input_mode : int
            Input mode (1 for database, 2 for CSV files)
        """
        self.config = load_data_config()
        self.logger = setup_logging(self.config, 'DataPuller')
        self.input_dir = (os.path.dirname(__file__))
        os.makedirs(self.input_dir, exist_ok=True)

        # Set up database connection for Vertica
        vertica_creds = get_creds(self.config['database']['vertica']['credentials_file'])
        self.vertica_conn_info = {
            'host': self.config['database']['vertica']['host'],
            'port': self.config['database']['vertica']['port'],
            'database': self.config['database']['vertica']['database'],
            'user': vertica_creds[0],
            'password': vertica_creds[1],
            'use_prepared_statements': False,
            'autocommit': True,
            'tlsmode': 'disable'
        }
            

    def read_sql_file(self, file_name: str) -> str:
        """
        Read SQL query from a file.
        
        Parameters
        ----------
        file_name : str
            Name of the SQL file to read
            
        Returns
        -------
        str
            SQL query as a string
        """
        sql_dir = os.path.join(os.path.dirname(__file__), 'sql')
        file_path = os.path.join(sql_dir, file_name)
        with open(file_path, 'r') as file:
            return file.read()

    def execute_vertica_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a query against Vertica database.
        
        Parameters
        ----------
        query : str
            SQL query to execute
        params : Optional[Dict[str, Any]], default=None
            Parameters to format into the query
            
        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame
        """
        self.logger.debug(f"Executing Vertica query: {query[:100]}...")
        with vertica_python.connect(**self.vertica_conn_info) as connection:
            with connection.cursor() as cursor:
                if params:
                    query = query.format(**params)
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)

    def get_container_data(self, fc: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Retrieve container data from database or CSV.
        
        Parameters
        ----------
        fc : str
            Fulfillment center ID
        start_time : str
            Start time for data retrieval
        end_time : str
            End time for data retrieval
            
        Returns
        -------
        pd.DataFrame
            Container data as a DataFrame
        """
        # Get data from database
        query = self.read_sql_file('container_data.sql')
        result_df = self.execute_vertica_query(query, {'fc': fc, 'start_time': start_time, 'end_time': end_time})
        
        # Comment out below - for testing only
        #result_df = result_df.head(1000)
        
        result_df.to_csv(os.path.join(self.input_dir, "container_data.csv"), index=False)
        
        return result_df

    def get_slotbook_data(self, fc: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve slotbook data from database or CSV.
        
        Parameters
        ----------
        fc : str
            Fulfillment center ID
            
        Returns
        -------
        pd.DataFrame
            Slotbook data as a DataFrame
        """
        # Get data from database
        query = self.read_sql_file('slotbook_data.sql')
        result_df = self.execute_vertica_query(query, {'fc': fc, 'start_date': start_date, 'end_date': end_date})
        result_df.to_csv(os.path.join(self.input_dir, "slotbook_data.csv"), index=False)

    
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

