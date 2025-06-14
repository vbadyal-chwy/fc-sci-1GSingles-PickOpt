"""
Enhanced Database Manager for Pick Optimization Simulation.

This module extends the existing SQLiteDataImporter with simulation-specific
functionality for managing TF/TA data and enhanced output tables.
"""
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import the existing SQLiteDataImporter
import sys
sys.path.append(str(Path(__file__).parent.parent / "external_data"))
from sqlite_importer import SQLiteDataImporter


class SimulationDBManager(SQLiteDataImporter):
    """Enhanced database manager with execution-scoped operations for simulation."""
    
    def __init__(self, db_path: str, execution_id: str):
        """
        Initialize with database path and execution ID for scoped operations.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database
        execution_id : str
            Execution ID for scoped operations
        """
        super().__init__(db_path)
        self.execution_id = execution_id
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"SimulationDBManager initialized for execution: {execution_id}")
    
    # ========================================================================
    # TF Data Management Methods
    # ========================================================================
    
    def get_containers_for_tf(self, planning_datetime: datetime, 
                             apply_filters: bool = True) -> pd.DataFrame:
        """
        Get containers ready for TF processing at the given planning datetime.
        
        Parameters
        ----------
        planning_datetime : datetime
            Planning timestamp for filtering containers
        apply_filters : bool
            Whether to apply TF-ready filters (arrive_datetime, not released, etc.)
            
        Returns
        -------
        pd.DataFrame
            Container data ready for TF
        """
        self.logger.info(f"Retrieving containers for TF at {planning_datetime}")
        
        try:
            with self._get_connection() as conn:
                if apply_filters:
                    # Get containers that have arrived but not yet released
                    query = """
                        SELECT * FROM containers 
                        WHERE execution_id = ? 
                        AND arrive_datetime <= ?
                        AND (released_flag = 0 OR released_flag IS NULL)
                        ORDER BY priority DESC, arrive_datetime ASC
                    """
                    params = (self.execution_id, planning_datetime)
                else:
                    # Get all containers for this execution
                    query = """
                        SELECT * FROM containers 
                        WHERE execution_id = ?
                        ORDER BY priority DESC, arrive_datetime ASC
                    """
                    params = (self.execution_id,)
                
                containers_df = pd.read_sql_query(query, conn, params=params)
                
                # Convert datetime columns
                if 'arrive_datetime' in containers_df.columns:
                    containers_df['arrive_datetime'] = pd.to_datetime(containers_df['arrive_datetime'])
                if 'original_promised_pull_datetime' in containers_df.columns:
                    containers_df['original_promised_pull_datetime'] = pd.to_datetime(containers_df['original_promised_pull_datetime'])
                
                self.logger.info(f"Retrieved {len(containers_df)} containers for TF")
                return containers_df
                
        except Exception as e:
            self.logger.error(f"Failed to get containers for TF: {e}")
            return pd.DataFrame()
    

    
    def store_tf_slack_calculations(self, slack_df: pd.DataFrame, 
                                   planning_datetime: datetime) -> bool:
        """
        Store TF slack calculations in the database.
        
        Parameters
        ----------
        slack_df : pd.DataFrame
            Slack calculation results
        planning_datetime : datetime
            Planning timestamp for this calculation
            
        Returns
        -------
        bool
            True if successful
        """
        if slack_df.empty:
            self.logger.warning("No slack data to store")
            return True
        
        self.logger.info(f"Storing {len(slack_df)} slack calculations for {planning_datetime}")
        
        try:
            # Add required columns
            slack_df = slack_df.copy()
            slack_df['execution_id'] = self.execution_id
            slack_df['planning_datetime'] = planning_datetime
            slack_df['created_at'] = datetime.now()
            
            with self._get_connection() as conn:
                slack_df.to_sql(
                    'tf_container_slack', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info("Slack calculations stored successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store slack calculations: {e}")
            return False
    
    def store_tf_target_calculations(self, target_df: pd.DataFrame, 
                                    planning_datetime: datetime) -> bool:
        """
        Store TF target calculations in the database.
        
        Parameters
        ----------
        target_df : pd.DataFrame
            Target calculation results
        planning_datetime : datetime
            Planning timestamp for this calculation
            
        Returns
        -------
        bool
            True if successful
        """
        if target_df.empty:
            self.logger.warning("No target data to store")
            return True
        
        self.logger.info(f"Storing target calculations for {planning_datetime}")
        
        try:
            # Add required columns
            target_df = target_df.copy()
            target_df['execution_id'] = self.execution_id
            target_df['planning_datetime'] = planning_datetime
            target_df['created_at'] = datetime.now()
            
            with self._get_connection() as conn:
                target_df.to_sql(
                    'tf_container_target', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info("Target calculations stored successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store target calculations: {e}")
            return False
    
    def store_tf_clustering_results(self, clustering_df: pd.DataFrame, 
                                   metadata_df: pd.DataFrame,
                                   planning_datetime: datetime) -> bool:
        """
        Store TF clustering results and metadata in the database.
        
        Parameters
        ----------
        clustering_df : pd.DataFrame
            Container clustering assignments
        metadata_df : pd.DataFrame
            Clustering metadata
        planning_datetime : datetime
            Planning timestamp
            
        Returns
        -------
        bool
            True if successful
        """
        self.logger.info(f"Storing TF clustering results for {planning_datetime}")
        
        try:
            with self._get_connection() as conn:
                # Store clustering assignments
                if not clustering_df.empty:
                    clustering_df = clustering_df.copy()
                    clustering_df['execution_id'] = self.execution_id
                    clustering_df['planning_datetime'] = planning_datetime
                    clustering_df['created_at'] = datetime.now()
                    
                    clustering_df.to_sql(
                        'tf_container_clustering', 
                        conn, 
                        if_exists='append', 
                        index=False
                    )
                
                # Store clustering metadata
                if not metadata_df.empty:
                    metadata_df = metadata_df.copy()
                    metadata_df['execution_id'] = self.execution_id
                    metadata_df['planning_datetime'] = planning_datetime
                    metadata_df['created_at'] = datetime.now()
                    
                    metadata_df.to_sql(
                        'tf_clustering_metadata', 
                        conn, 
                        if_exists='append', 
                        index=False
                    )
                
                self.logger.info("TF clustering results stored successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store TF clustering results: {e}")
            return False
    
    def store_tf_tour_formation(self, tours_df: pd.DataFrame, 
                               planning_datetime: datetime) -> bool:
        """
        Store TF tour formation results in the database.
        
        Parameters
        ----------
        tours_df : pd.DataFrame
            Tour formation results
        planning_datetime : datetime
            Planning timestamp
            
        Returns
        -------
        bool
            True if successful
        """
        if tours_df.empty:
            self.logger.warning("No tour formation data to store")
            return True
        
        self.logger.info(f"Storing {len(tours_df)} tour formation records")
        
        try:
            # Add required columns
            tours_df = tours_df.copy()
            tours_df['execution_id'] = self.execution_id
            tours_df['planning_datetime'] = planning_datetime
            tours_df['created_at'] = datetime.now()
            
            with self._get_connection() as conn:
                # Check for existing records and remove duplicates
                existing_query = """
                    SELECT COUNT(*) FROM tf_tour_formation 
                    WHERE execution_id = ? AND planning_datetime = ?
                """
                existing_count = conn.execute(existing_query, (self.execution_id, planning_datetime)).fetchone()[0]
                
                if existing_count > 0:
                    self.logger.warning(f"Found {existing_count} existing tour formation records for {planning_datetime}, skipping insertion")
                    return True
                
                # Handle NULL pick_location values
                tours_df['pick_location'] = tours_df['pick_location'].fillna('UNKNOWN')
                
                tours_df.to_sql(
                    'tf_tour_formation', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info("Tour formation data stored successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store tour formation data: {e}")
            return False
    
    def clear_ready_tours_for_execution(self) -> bool:
        """
        Clear all ready-to-release tours for the current execution.
        
        This removes all records from ready_to_release_tours table for the current 
        execution_id, effectively resetting the ready pool.
        
        Returns
        -------
        bool
            True if successful
        """
        self.logger.info(f"Clearing all ready tours for execution: {self.execution_id}")
        
        try:
            with self._get_connection() as conn:
                # Delete all records for this execution
                delete_query = """
                    DELETE FROM ready_to_release_tours 
                    WHERE execution_id = ?
                """
                cursor = conn.execute(delete_query, (self.execution_id,))
                rows_deleted = cursor.rowcount
                
                self.logger.info(f"Cleared {rows_deleted} ready tour records for execution {self.execution_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear ready tours for execution: {e}")
            return False
    
    def add_tours_to_ready_pool(self, ready_tours_df: pd.DataFrame, 
                               created_at_datetime: datetime) -> bool:
        """
        Add tours to the ready-to-release pool.
        
        This method first clears all existing ready tours for the current execution,
        then adds the new tours. This ensures a clean reset of the ready pool
        for each TF run.
        
        Parameters
        ----------
        ready_tours_df : pd.DataFrame
            Tours ready to be released
        created_at_datetime : datetime
            When these tours were created (TF planning_datetime)
            
        Returns
        -------
        bool
            True if successful
        """
        # First, clear existing ready tours for this execution
        if not self.clear_ready_tours_for_execution():
            self.logger.error("Failed to clear existing ready tours")
            return False
        
        if ready_tours_df.empty:
            self.logger.warning("No tours to add to ready pool")
            return True
        
        self.logger.info(f"Adding {len(ready_tours_df)} lines to ready-to-release pool")
        
        try:
            # Add required columns
            ready_tours_df = ready_tours_df.copy()
            ready_tours_df['execution_id'] = self.execution_id
            ready_tours_df['created_at_datetime'] = created_at_datetime
            ready_tours_df['archived_flag'] = False
            ready_tours_df['created_at'] = datetime.now()
            
            with self._get_connection() as conn:
                ready_tours_df.to_sql(
                    'ready_to_release_tours', 
                    conn, 
                    if_exists='append', 
                    index=False
                )
                
                self.logger.info("Tours added to ready pool successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add tours to ready pool: {e}")
            return False
    
    def update_tf_clustering_metadata_with_solve_metrics(self, solve_metrics_df: pd.DataFrame, 
                                                        planning_datetime: datetime) -> bool:
        """
        Update tf_clustering_metadata table with additional solve metrics from cluster solutions.
        
        Parameters
        ----------
        solve_metrics_df : pd.DataFrame
            Combined solve metrics data from cluster_{id}_solve_metrics.csv files
        planning_datetime : datetime
            Planning timestamp
            
        Returns
        -------
        bool
            True if successful
        """
        if solve_metrics_df.empty:
            self.logger.warning("No solve metrics data to update")
            return True
        
        self.logger.info(f"Updating clustering metadata with solve metrics for {len(solve_metrics_df)} clusters")
        
        try:
            with self._get_connection() as conn:
                for _, row in solve_metrics_df.iterrows():
                    conn.execute("""
                        UPDATE tf_clustering_metadata 
                        SET solve_time = ?, objective_value = ?, total_distance = ?, 
                            total_slack = ?, actual_tour_count = ?
                        WHERE execution_id = ? AND planning_datetime = ? AND cluster_id = ?
                    """, (
                        row.get('solve_time'), row.get('objective_value'), 
                        row.get('total_distance'), row.get('total_slack'), 
                        row.get('actual_tour_count'),
                        self.execution_id, planning_datetime, row.get('cluster_id')
                    ))
                
                self.logger.info("Clustering metadata updated with solve metrics successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update clustering metadata with solve metrics: {e}")
            return False

    def update_container_release_status(self, tour_ids: List[str], planning_datetime: datetime) -> dict:
        """
        Update container release status for containers in the specified tours.
        
        Sets released_flag=TRUE, tour_id, and release_datetime for all containers
        that are part of the released tours. This method should be called after
        tours have been archived in the ready_to_release_tours table.
        
        Parameters
        ----------
        tour_ids : List[str]
            List of tour IDs that were released by TA
        planning_datetime : datetime
            Planning timestamp when tours were released
            
        Returns
        -------
        dict
            Dictionary with update results:
            - 'updated': Number of containers actually updated
            - 'expected': Number of containers expected to be updated
            - 'tour_ids': List of tour IDs processed
            - 'error': Error message if operation failed
            
        Raises
        ------
        sqlite3.Error
            If database update fails
        """
        if not tour_ids:
            self.logger.warning("No tour IDs provided for container release status update")
            return {'updated': 0, 'expected': 0, 'tour_ids': []}
        
        self.logger.info(f"Updating container release status for {len(tour_ids)} tours")
        
        try:
            with self._get_connection() as conn:
                # First, get containers that should be updated
                placeholders = ','.join(['?' for _ in tour_ids])
                query_containers = f"""
                    SELECT DISTINCT container_id, tour_id
                    FROM ready_to_release_tours 
                    WHERE execution_id = ? AND tour_id IN ({placeholders})
                    AND archived_flag = 1
                """
                
                containers_df = pd.read_sql_query(
                    query_containers, 
                    conn, 
                    params=[self.execution_id] + tour_ids
                )
                
                expected_updates = len(containers_df)
                
                if expected_updates == 0:
                    self.logger.warning("No containers found for the specified tour IDs")
                    return {'updated': 0, 'expected': 0, 'tour_ids': tour_ids}
                
                self.logger.info(f"Found {expected_updates} containers to update for released tours")
                
                # Update containers table
                update_query = f"""
                    UPDATE containers 
                    SET released_flag = 1,
                        tour_id = (
                            SELECT tour_id FROM ready_to_release_tours r 
                            WHERE r.execution_id = containers.execution_id 
                            AND r.container_id = containers.container_id 
                            AND r.tour_id IN ({placeholders})
                            AND r.archived_flag = 1
                            LIMIT 1
                        ),
                        release_datetime = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE execution_id = ? 
                    AND container_id IN (
                        SELECT DISTINCT container_id FROM ready_to_release_tours 
                        WHERE execution_id = ? AND tour_id IN ({placeholders})
                        AND archived_flag = 1
                    )
                """
                
                # Parameters: planning_datetime, execution_id, execution_id, tour_ids...
                params = [planning_datetime, self.execution_id, self.execution_id] + tour_ids
                
                cursor = conn.execute(update_query, params)
                updated_count = cursor.rowcount
                
                self.logger.info(f"Updated release status for {updated_count} containers (expected {expected_updates})")
                
                if updated_count != expected_updates:
                    self.logger.warning(f"Container update mismatch: updated {updated_count}, expected {expected_updates}")
                
                return {
                    'updated': updated_count,
                    'expected': expected_updates,
                    'tour_ids': tour_ids
                }
                
        except Exception as e:
            self.logger.error(f"Failed to update container release status: {e}")
            return {'updated': 0, 'expected': 0, 'tour_ids': tour_ids, 'error': str(e)}

    