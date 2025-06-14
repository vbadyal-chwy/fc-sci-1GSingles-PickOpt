"""
External data extraction module for Pick Optimization.

This module handles extracting data from Snowflake and Vertica using configured SQL queries
and the existing db_util.py infrastructure.
"""

import sys
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Add the parent directories to Python path to import utilities
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))

from simulator.utils.db_util import DBUtil


class ExternalDataExtractor:
    """Handles data extraction from Snowflake and Vertica using configured SQL queries."""
    
    def __init__(self, config_path: str):
        """
        Initialize with configuration file path.
        
        Parameters
        ----------
        config_path : str
            Path to the sim_config.yaml file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database utility
        self.db_util = DBUtil(config_path)
        
        # SQL files directory
        self.sql_dir = project_root / "simulator" / "sql" / "snowflake"
        
        self.logger.info("ExternalDataExtractor initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _read_sql_file(self, filename: str) -> str:
        """Read SQL query from file."""
        sql_path = self.sql_dir / filename
        try:
            with open(sql_path, 'r') as f:
                sql_content = f.read()
            self.logger.debug(f"SQL file loaded: {filename}")
            return sql_content
        except FileNotFoundError:
            self.logger.error(f"SQL file not found: {sql_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading SQL file {filename}: {e}")
            raise
    
    def extract_container_master(self, wh_id: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Extract container master data from Snowflake.
        
        Parameters
        ----------
        wh_id : str
            Warehouse ID (e.g., 'AVP1')
        start_time : str
            Start time for data extraction (format: 'YYYY-MM-DD HH:MM:SS')
        end_time : str
            End time for data extraction (format: 'YYYY-MM-DD HH:MM:SS')
            
        Returns
        -------
        pd.DataFrame
            Container master data
        """
        self.logger.info(f"Extracting container master data for {wh_id} from {start_time} to {end_time}")
        
        # Read SQL query
        sql_query = self._read_sql_file("container_master.sql")
        
        # Create parameter list based on order of %s placeholders in SQL
        query_params = [
            wh_id,        # warehouse_config where clause
            wh_id,        # autobatch_release where clause  
            start_time,   # autobatch_release date range start (with -4 offset)
            end_time,     # autobatch_release date range end (with +2 offset)
            wh_id,        # containers_charged_during_period where clause
            start_time,   # containers_charged_during_period arrive_date range start
            end_time,     # containers_charged_during_period arrive_date range end
            wh_id,        # containers_charged_before_batched_after where clause
            start_time,   # containers_charged_before_batched_after arrive_date comparison
            start_time    # containers_charged_before_batched_after autobatched_date_local comparison
        ]
        
        self.logger.info(f"Executing container_master.sql with parameters: {query_params}")
        
        # Execute query with parameters
        result_df = self.db_util.execute_snowflake_query(sql_query, params=query_params)
        
        self.logger.info(f"Container master data extracted: {len(result_df)} rows")
        return result_df
    
    def extract_container_details(self, wh_id: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Extract container pick-level details from Snowflake.
        
        Parameters
        ----------
        wh_id : str
            Warehouse ID (e.g., 'AVP1')
        start_time : str
            Start time for data extraction (format: 'YYYY-MM-DD HH:MM:SS')
        end_time : str
            End time for data extraction (format: 'YYYY-MM-DD HH:MM:SS')
            
        Returns
        -------
        pd.DataFrame
            Container pick-level details
        """
        self.logger.info(f"Extracting container details for {wh_id} from {start_time} to {end_time}")
        
        # Read SQL query
        sql_query = self._read_sql_file("container_detail.sql")
        
        # Create parameter list based on order of %s placeholders in SQL
        query_params = [
            wh_id,        # warehouse_config where clause
            wh_id,        # autobatch_release where clause
            start_time,   # autobatch_release date range start (with -4 offset)
            end_time,     # autobatch_release date range end (with +2 offset)
            wh_id,        # eligible_containers first union where clause
            start_time,   # eligible_containers first union arrive_date range start
            end_time,     # eligible_containers first union arrive_date range end
            wh_id,        # eligible_containers second union where clause
            start_time,   # eligible_containers second union arrive_date comparison
            start_time    # eligible_containers second union autobatched_date_local comparison
        ]
        
        self.logger.info(f"Executing container_detail.sql with parameters: {query_params}")
        
        # Execute query with parameters
        result_df = self.db_util.execute_snowflake_query(sql_query, params=query_params)
        
        self.logger.info(f"Container details extracted: {len(result_df)} rows")
        return result_df
    
    def extract_slotbook_data(self, wh_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract slotbook/inventory data from Snowflake.
        
        Parameters
        ----------
        wh_id : str
            Warehouse ID (e.g., 'AVP1')
        start_date : str
            Start date for data extraction (format: 'YYYY-MM-DD')
        end_date : str
            End date for data extraction (format: 'YYYY-MM-DD')
            
        Returns
        -------
        pd.DataFrame
            Slotbook/inventory data
        """
        self.logger.info(f"Extracting slotbook data for {wh_id} from {start_date} to {end_date}")
        
        # Read SQL query
        sql_query = self._read_sql_file("slotbook_data.sql")
        
        # Create parameter list based on order of %s placeholders in SQL
        query_params = [
            wh_id,        # warehouse where clause
            start_date,   # date range start
            end_date      # date range end
        ]
        
        self.logger.info(f"Executing slotbook_data.sql with parameters: {query_params}")
        
        # Execute query with parameters
        result_df = self.db_util.execute_snowflake_query(sql_query, params=query_params)
        
        self.logger.info(f"Slotbook data extracted: {len(result_df)} rows")
        return result_df
    
    def extract_labor_data(self, wh_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract labor data from Vertica.
        
        Parameters
        ----------
        wh_id : str
            Warehouse ID (e.g., 'AVP1')
        start_date : str
            Start date for data extraction (format: 'YYYY-MM-DD')
        end_date : str
            End date for data extraction (format: 'YYYY-MM-DD')
            
        Returns
        -------
        pd.DataFrame
            Labor data
        """
        self.logger.info(f"Extracting labor data for {wh_id} from {start_date} to {end_date}")
        
        # Read SQL query from vertica directory
        sql_path = self.sql_dir.parent / "vertica" / "labor_data.sql"
        try:
            with open(sql_path, 'r') as f:
                sql_content = f.read()
            self.logger.debug(f"Labor SQL file loaded: labor_data.sql")
        except FileNotFoundError:
            self.logger.error(f"Labor SQL file not found: {sql_path}")
            raise
        
        # Replace parameters in SQL query (3 instances of wh_id, plus start and end dates)
        formatted_query = sql_content.replace('%s', f"'{wh_id}'", 3)  # Replace first 3 %s with wh_id
        formatted_query = formatted_query.replace('%s', f"'{start_date}'", 1)  # Replace 4th %s with start_date  
        formatted_query = formatted_query.replace('%s', f"'{end_date}'", 1)   # Replace 5th %s with end_date
        
        # Execute query using Vertica connection (assuming db_util supports it)
        result_df = self.db_util.execute_vertica_query(formatted_query)
        
        self.logger.info(f"Labor data extracted: {len(result_df)} rows")
        return result_df

    def extract_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract all required data using configuration parameters.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all extracted data:
            - 'container_master': Container master data
            - 'container_details': Container pick-level details  
            - 'slotbook': Slotbook/inventory data
            - 'labor_data': Labor data from Vertica
        """
        # Get parameters from configuration
        global_config = self.config.get('global', {})
        wh_id = global_config.get('wh_id')
        start_time = global_config.get('start_time')
        end_time = global_config.get('end_time')
        
        if not all([wh_id, start_time, end_time]):
            raise ValueError("Missing required configuration: wh_id, start_time, or end_time")
        
        # Extract date portion for slotbook and labor queries
        start_date = start_time.split(' ')[0]
        end_date = end_time.split(' ')[0]
        
        self.logger.info(f"Starting data extraction for {wh_id}")
        
        # Extract all data
        data = {}
        
        try:
            data['container_master'] = self.extract_container_master(wh_id, start_time, end_time)
            data['container_details'] = self.extract_container_details(wh_id, start_time, end_time)
            data['slotbook'] = self.extract_slotbook_data(wh_id, start_date, end_date)
            data['labor_data'] = self.extract_labor_data(wh_id, start_date, end_date)
            
            self.logger.info("Data extraction completed successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise 