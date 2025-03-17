"""
Data retrieval module for pick optimization.

This module provides functionality to retrieve data from various sources,
including databases and CSV files.
"""
import os
import pandas as pd
import vertica_python
from typing import Dict, Any, Optional

from config.config_loader import load_config, get_creds
from utils.logging_config import setup_logging


class DataPuller:
    """
    Data retrieval class for pick planning optimization.
    
    This class handles retrieving container and slotbook data from
    either a database or CSV files based on the configured input mode.
    """
    
    def __init__(self, config_path: str, input_mode: int):
        """
        Initialize the DataPuller with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        input_mode : int
            Input mode (1 for database, 2 for CSV files)
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config, 'DataPuller')
        self.config_path = config_path
        self.input_mode = input_mode
        self.project_root = os.path.dirname(os.path.dirname(self.config_path))
        self.input_dir = os.path.join(self.project_root, "input")
        os.makedirs(self.input_dir, exist_ok=True)

        if input_mode == 1:
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
        if self.input_mode == 1:
            # Get data from database
            query = self.read_sql_file('container_data.sql')
            result_df = self.execute_vertica_query(query, {'fc': fc, 'start_time': start_time, 'end_time': end_time})
            result_df.to_csv(os.path.join(self.input_dir, "container_data.csv"), index=False)
        else:
            # Get data from CSV
            csv_path = os.path.join(self.input_dir, "container_data-manual.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Container data file not found at {csv_path}")
            result_df = pd.read_csv(csv_path)
            # Convert datetime columns to pd datetime objects
            result_df['arrive_datetime'] = pd.to_datetime(result_df['arrive_datetime'])
            result_df['cut_datetime'] = pd.to_datetime(result_df['cut_datetime'])
            result_df['pull_datetime'] = pd.to_datetime(result_df['pull_datetime'])
        return result_df

    def get_slotbook_data(self, fc: str) -> pd.DataFrame:
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
        if self.input_mode == 1:
            # Get data from database
            query = self.read_sql_file('slotbook_data.sql')
            result_df = self.execute_vertica_query(query, {'fc': fc})
            result_df.to_csv(os.path.join(self.input_dir, "slotbook_data.csv"), index=False)
        else:
            # Get data from CSV
            csv_path = os.path.join(self.input_dir, "slotbook_data-manual.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Slotbook data file not found at {csv_path}")
            result_df = pd.read_csv(csv_path)
        return result_df 