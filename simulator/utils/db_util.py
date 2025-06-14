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
from typing import Any, Optional, Tuple, Sequence
from pathlib import Path
from cryptography.fernet import Fernet
import snowflake.connector


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
    if not username and "cred.txt" in file_path: 
        try:
           
            username = os.path.basename(Path(file_path).resolve().parents[0])
        except IndexError:
            username = "unknown_user"
    return username, decrypted_credentials

class DBUtil:
    """Utility class for handling database connections and queries."""
    
    def __init__(self, config_file_path: str = "src/sim_config.yaml"):
        """
        Initialize DBUtil with a configuration file.

        Args:
            config_file_path (str): Path to the YAML configuration file.
        """
        self.config_file_path = config_file_path
        self.logger = logging.getLogger(__name__)

        self.vertica_conn_info = {}
        self.snowflake_conn_info = {}
        self._load_db_config_from_file()

    def _load_db_config_from_file(self):
        """Loads database configurations from the YAML file specified in config_file_path."""
        try:
            with open(self.config_file_path, 'r') as cf:
                config_data = yaml.safe_load(cf)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_file_path}")
            return
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration file {self.config_file_path}: {e}")
            return

        if 'database' not in config_data:
            self.logger.error("'database' section not found in configuration.")
            return

        db_config = config_data['database']

        # Load Vertica configuration
        if 'vertica' in db_config:
            vertica_cfg = db_config['vertica']
            try:
                user_env_var = os.getenv('USERNAME', os.getenv('USER', ''))
                creds_file = vertica_cfg['credentials_file'].replace('${USER}', user_env_var)
                
                # Regarding creds_file path being absolute or relative:
                # The get_creds function will attempt to resolve it. 
                # The primary concern is that the path correctly points to the credentials file.
                # No specific 'pass' logic needed here if get_creds handles path variations.
                # If '${USER}' was in the original path, it's assumed to be part of a base path that becomes absolute.
                # If not, and it's relative, it should be relative to where the script expects it (e.g. CWD or config file dir).
                # The current get_creds is simple; for truly robust relative path handling based on config file location,
                # creds_file path might need to be os.path.join(os.path.dirname(self.config_file_path), creds_file)
                # if it's detected as relative and not containing ${USER}.
                # For now, we rely on get_creds and the input path string being resolvable.

                username, password = get_creds(creds_file)
                self.vertica_conn_info = {
                    'host': vertica_cfg['host'],
                    'port': vertica_cfg['port'],
                    'database': vertica_cfg['database'],
                    'user': username,
                    'password': password,
                    'use_prepared_statements': vertica_cfg.get('use_prepared_statements', False),
                    'autocommit': vertica_cfg.get('autocommit', True),
                    'tlsmode': vertica_cfg.get('tlsmode', 'disable')
                }
                self.logger.info("Vertica configuration loaded.")
            except KeyError as e:
                self.logger.error(f"Missing key in Vertica configuration: {e}")
            except Exception as e:
                self.logger.error(f"Error loading Vertica credentials: {e}")

        # Load Snowflake configuration
        if 'snowflake' in db_config:
            snowflake_cfg = db_config['snowflake']
            try:
                self.snowflake_conn_info = {
                    'user': snowflake_cfg['user'],
                    'account': snowflake_cfg['account'],
                    'warehouse': snowflake_cfg['warehouse'],
                    'database': snowflake_cfg['database'],
                    'schema': snowflake_cfg['schema'],
                    # Optional parameters
                    'password': snowflake_cfg.get('password'), 
                    'authenticator': snowflake_cfg.get('authenticator'),
                    'role': snowflake_cfg.get('role')
                }
                self.logger.info("Snowflake configuration loaded.")
                
                authenticator = self.snowflake_conn_info.get('authenticator')
                password = self.snowflake_conn_info.get('password')

                if not authenticator or authenticator.lower() not in ['externalbrowser', 'oauth']:
                    if not password:
                        self.logger.warning("Snowflake password is not set and authenticator does not bypass password requirement. Connection may fail.")
                elif authenticator and authenticator.lower() == 'externalbrowser' and password:
                    self.logger.info("Snowflake is using 'externalbrowser' authenticator; provided password will be ignored by the connector.")

            except KeyError as e:
                self.logger.error(f"Missing key in Snowflake configuration: {e}")

    def read_sql_file(self, sql_file_path: str) -> str:
        """
        Read an SQL query from the specified file path.
        
        Parameters
        ----------
        sql_file_path : str
            The full path to the SQL file.
            
        Returns
        -------
        str
            SQL query as a string, or empty string if file not found.
        """
        try:
            with open(sql_file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            self.logger.error(f"SQL file not found at {sql_file_path}")
            return "" # Or raise an exception

    def execute_vertica_query(self, query: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """
        Execute a query against Vertica database.
        
        Parameters
        ----------
        query : str
            SQL query to execute
        params : Optional[Sequence[Any]], default=None
            Parameters to substitute into the query using DB-API style
            
        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame
        """
        if not self.vertica_conn_info.get('user') or not self.vertica_conn_info.get('password'):
            self.logger.error("Vertica connection info is not properly configured or missing credentials.")
            return pd.DataFrame()

        self.logger.debug(f"Executing Vertica query (first 100 chars): {query[:100]}...")
        try:
            with vertica_python.connect(**self.vertica_conn_info) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params if params else [])
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    self.logger.info(f"Vertica query executed. Rows fetched: {len(data)}")
                    return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"Error executing Vertica query: {e}")
            return pd.DataFrame()

    def execute_snowflake_query(self, query: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
        """
        Execute a query against Snowflake database.

        Parameters
        ----------
        query : str
            SQL query to execute
        params : Optional[Sequence[Any]], default=None
            Parameters to use with the query (e.g., for %s or ? placeholders)

        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame
        """
        if not self.snowflake_conn_info.get('user') or not self.snowflake_conn_info.get('account'):
            self.logger.error("Snowflake connection info is not properly configured (user/account missing).")
            return pd.DataFrame()

        authenticator = self.snowflake_conn_info.get('authenticator')
        password_present = self.snowflake_conn_info.get('password') is not None

        if not authenticator or authenticator.lower() not in ['externalbrowser', 'oauth']:
            if not password_present:
                self.logger.error("Snowflake password is required but not set in configuration, and no overriding authenticator is specified.")
                return pd.DataFrame()

        self.logger.debug(f"Executing Snowflake query (first 100 chars): {query[:100]}...")
        conn = None
        cursor = None
        try:
            conn = snowflake.connector.connect(**self.snowflake_conn_info)
            cursor = conn.cursor()
            cursor.execute(query, params) 
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            self.logger.info(f"Snowflake query executed. Rows fetched: {len(data)}")
            return pd.DataFrame(data, columns=columns)
        except snowflake.connector.errors.ProgrammingError as e:
            self.logger.error(f"Snowflake Programming Error: {e}")
            if "Incorrect username or password was specified" in str(e) or "Failed to connect" in str(e):
                self.logger.error("Please check your Snowflake username, password, and other connection details in the config file.")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error executing Snowflake query: {e}")
            return pd.DataFrame()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            self.logger.debug("Snowflake connection closed.")