"""
SQLite data import module for Pick Optimization.

This module handles importing transformed data into the SQLite database
with transaction safety and foreign key validation.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class SQLiteDataImporter:
    """Handles data import into SQLite database."""
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Verify database exists
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        self.logger.info(f"SQLiteDataImporter initialized with database: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign key constraints enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def create_execution_record(self, execution_id: str, wh_id: str, 
                               start_time: str, end_time: str,
                               tf_interval_minutes: int = 30,
                               ta_interval_minutes: int = 5,
                               tf_min_containers_in_backlog: int = 1000) -> bool:
        """
        Create execution record to track this data loading session.
        
        Parameters
        ----------
        execution_id : str
            Unique execution ID
        wh_id : str
            Warehouse ID
        start_time : str
            Start time for simulation
        end_time : str
            End time for simulation
        tf_interval_minutes : int
            Tour Formation interval in minutes
        ta_interval_minutes : int
            Tour Allocation interval in minutes
        tf_min_containers_in_backlog : int
            Minimum containers required in backlog for TF
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        self.logger.info(f"Creating execution record for {execution_id}")
        
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO executions (
                        execution_id, wh_id, start_time, end_time,
                        tf_interval_minutes, ta_interval_minutes, tf_min_containers_in_backlog,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    execution_id, wh_id, start_time, end_time,
                    tf_interval_minutes, ta_interval_minutes, tf_min_containers_in_backlog,
                    'INITIALIZED', datetime.now(), datetime.now()
                ))
                
                self.logger.info(f"Execution record created: {execution_id}")
                return True
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                self.logger.warning(f"Execution {execution_id} already exists, updating status")
                return self._update_execution_status(execution_id, 'loading')
            else:
                self.logger.error(f"Failed to create execution record: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to create execution record: {e}")
            return False
    
    def _update_execution_status(self, execution_id: str, status: str) -> bool:
        """Update execution status."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE executions 
                    SET status = ?, updated_at = ?
                    WHERE execution_id = ?
                """, (status, datetime.now(), execution_id))
                
                self.logger.info(f"Execution status updated: {execution_id} -> {status}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update execution status: {e}")
            return False
    
    def import_containers(self, containers_df: pd.DataFrame) -> bool:
        """
        Import containers data into the database.
        
        Parameters
        ----------
        containers_df : pd.DataFrame
            Container data to import
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if containers_df.empty:
            self.logger.warning("No container data to import")
            return True
        
        self.logger.info(f"Importing {len(containers_df)} containers")
        
        try:
            with self._get_connection() as conn:
                # Use pandas to_sql for efficient bulk insert
                containers_df.to_sql(
                    'containers', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info(f"Containers imported successfully: {len(containers_df)} rows")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import containers: {e}")
            return False
    
    def import_container_details(self, details_df: pd.DataFrame) -> bool:
        """
        Import container details data into the database.
        
        Parameters
        ----------
        details_df : pd.DataFrame
            Container details data to import
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if details_df.empty:
            self.logger.warning("No container details data to import")
            return True
        
        self.logger.info(f"Importing {len(details_df)} container details")
        
        try:
            with self._get_connection() as conn:
                # Use pandas to_sql for efficient bulk insert
                details_df.to_sql(
                    'container_details', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info(f"Container details imported successfully: {len(details_df)} rows")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import container details: {e}")
            return False
    
    def import_slotbook(self, slotbook_df: pd.DataFrame) -> bool:
        """
        Import slotbook data into the database.
        
        Parameters
        ----------
        slotbook_df : pd.DataFrame
            Slotbook data to import
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if slotbook_df.empty:
            self.logger.warning("No slotbook data to import")
            return True
        
        self.logger.info(f"Importing {len(slotbook_df)} slotbook records")
        
        try:
            with self._get_connection() as conn:
                # Use pandas to_sql for efficient bulk insert
                slotbook_df.to_sql(
                    'slotbook', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info(f"Slotbook data imported successfully: {len(slotbook_df)} rows")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import slotbook data: {e}")
            return False
    
    def import_labor_data(self, labor_df: pd.DataFrame) -> bool:
        """
        Import labor data into the database.
        
        Parameters
        ----------
        labor_df : pd.DataFrame
            Labor data to import
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if labor_df.empty:
            self.logger.warning("No labor data to import")
            return True
        
        self.logger.info(f"Importing {len(labor_df)} labor records")
        
        try:
            with self._get_connection() as conn:
                # Use pandas to_sql for efficient bulk insert
                labor_df.to_sql(
                    'labor_data', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info(f"Labor data imported successfully: {len(labor_df)} rows")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to import labor data: {e}")
            return False
    
    def import_all_data(self, transformed_data: Dict[str, pd.DataFrame], 
                       execution_id: str, wh_id: str, 
                       start_time: str, end_time: str,
                       tf_interval_minutes: int,
                       ta_interval_minutes: int,
                       tf_min_containers_in_backlog: int = 1000) -> bool:
        """
        Import all transformed data into the database with transaction safety.
        
        Parameters
        ----------
        transformed_data : Dict[str, pd.DataFrame]
            Dictionary containing all transformed data
        execution_id : str
            Unique execution ID
        wh_id : str
            Warehouse ID
        start_time : str
            Start time for data extraction
        end_time : str
            End time for data extraction
            
        Returns
        -------
        bool
            True if all imports successful, False otherwise
        """
        self.logger.info(f"Starting bulk data import for execution: {execution_id}")
        
        # Create execution record first
        if not self.create_execution_record(execution_id, wh_id, start_time, end_time,
                                           tf_interval_minutes, ta_interval_minutes, tf_min_containers_in_backlog):
            self.logger.error("Failed to create execution record, aborting import")
            return False
        
        # Import all data in sequence
        success = True
        
        # Import containers first (required for foreign key references)
        if 'containers' in transformed_data:
            if not self.import_containers(transformed_data['containers']):
                success = False
        
        # Import container details
        if 'container_details' in transformed_data and success:
            if not self.import_container_details(transformed_data['container_details']):
                success = False
        
        # Import slotbook data
        if 'slotbook' in transformed_data and success:
            if not self.import_slotbook(transformed_data['slotbook']):
                success = False
        
        # Import labor data
        if 'labor_data' in transformed_data and success:
            if not self.import_labor_data(transformed_data['labor_data']):
                success = False
        
        # Update execution status
        final_status = 'completed' if success else 'failed'
        self._update_execution_status(execution_id, final_status)
        
        if success:
            self.logger.info(f"Bulk data import completed successfully for execution: {execution_id}")
        else:
            self.logger.error(f"Bulk data import failed for execution: {execution_id}")
        
        return success
    
    def cleanup_execution_data(self, execution_id: str) -> bool:
        """
        Remove all data for a specific execution (for cleanup/testing).
        
        Parameters
        ----------
        execution_id : str
            Execution ID to clean up
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        self.logger.info(f"Cleaning up data for execution: {execution_id}")
        
        try:
            with self._get_connection() as conn:
                # Delete in reverse order to respect foreign key constraints
                conn.execute("DELETE FROM labor_data WHERE execution_id = ?", (execution_id,))
                conn.execute("DELETE FROM slotbook WHERE execution_id = ?", (execution_id,))
                conn.execute("DELETE FROM container_details WHERE execution_id = ?", (execution_id,))
                conn.execute("DELETE FROM containers WHERE execution_id = ?", (execution_id,))
                conn.execute("DELETE FROM executions WHERE execution_id = ?", (execution_id,))
                
                self.logger.info(f"Cleanup completed for execution: {execution_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Cleanup failed for execution {execution_id}: {e}")
            return False
    
    def get_import_summary(self, execution_id: str) -> Dict[str, Any]:
        """
        Get summary of imported data for an execution.
        
        Parameters
        ----------
        execution_id : str
            Execution ID to summarize
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        self.logger.info(f"Getting import summary for execution: {execution_id}")
        
        try:
            with self._get_connection() as conn:
                # Get execution info
                exec_result = conn.execute("""
                    SELECT wh_id, status, created_at, updated_at
                    FROM executions 
                    WHERE execution_id = ?
                """, (execution_id,)).fetchone()
                
                if not exec_result:
                    return {"error": f"Execution {execution_id} not found"}
                
                # Get counts for each table
                containers_count = conn.execute("""
                    SELECT COUNT(*) FROM containers WHERE execution_id = ?
                """, (execution_id,)).fetchone()[0]
                
                details_count = conn.execute("""
                    SELECT COUNT(*) FROM container_details WHERE execution_id = ?
                """, (execution_id,)).fetchone()[0]
                
                slotbook_count = conn.execute("""
                    SELECT COUNT(*) FROM slotbook WHERE execution_id = ?
                """, (execution_id,)).fetchone()[0]
                
                labor_count = conn.execute("""
                    SELECT COUNT(*) FROM labor_data WHERE execution_id = ?
                """, (execution_id,)).fetchone()[0]
                
                summary = {
                    "execution_id": execution_id,
                    "wh_id": exec_result[0],
                    "status": exec_result[1],
                    "created_at": exec_result[2],
                    "updated_at": exec_result[3],
                    "counts": {
                        "containers": containers_count,
                        "container_details": details_count,
                        "slotbook": slotbook_count,
                        "labor_data": labor_count
                    }
                }
                
                self.logger.info(f"Import summary retrieved for execution: {execution_id}")
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get import summary: {e}")
            return {"error": str(e)} 