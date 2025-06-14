"""
Database-driven Tour Formation input generation utilities.

This module provides functions to create TF inputs directly from the SQLite database
instead of flat files, enabling seamless integration with the simulator workflow.
"""

import logging
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil


def create_tf_inputs_from_database(
    execution_id: str,
    planning_timestamp: datetime,
    db_path: Path,
    wh_id: str,
    base_input_dir: Path,
    tour_formation_config_path: Path,
    logger: Optional[logging.Logger] = None 
) -> Path:
    """
    Create TF inputs from database for a single planning timestamp.
    
    Args:
        execution_id: Database execution ID to filter data
        planning_timestamp: Planning timestamp for filtering containers and slotbook
        db_path: Path to SQLite database
        wh_id: Warehouse ID from config
        base_input_dir: Base directory for TF inputs
        tour_formation_config_path: Path to TF config file
        logger: Optional logger instance
        
    Returns:
        Path to created TF input directory
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Creating TF inputs from database for {planning_timestamp}")
    logger.debug(f"Execution ID: {execution_id}, Warehouse: {wh_id}")
    
    # Create timestamped directory
    tf_timestamp_str = planning_timestamp.strftime('%Y%m%d_%H%M%S')
    tf_input_dir = base_input_dir / wh_id / tf_timestamp_str
    tf_input_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Query enhanced container data
            logger.debug("Querying container data from database")
            container_query = """
            SELECT c.wh_id, c.container_id, c.priority, c.arrive_datetime, 
                   c.original_promised_pull_datetime as pull_datetime, cd.item_number, 
                   cd.planned_quantity as pick_quantity, cd.pick_location as wms_pick_location, cd.aisle_sequence, cd.picking_flow_as_int
            FROM containers c 
            JOIN container_details cd ON c.wh_id = cd.wh_id AND c.container_id = cd.container_id
            WHERE c.execution_id = ? 
            AND c.wh_id = ?
            AND c.arrive_datetime <= ?
            AND c.released_flag = 0
            """
            
            containers_df = pd.read_sql_query(
                container_query, 
                conn, 
                params=[execution_id, wh_id, planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')]
            )
            
            logger.info(f"Retrieved {len(containers_df)} container records")
            
            # Query slotbook data
            logger.debug("Querying slotbook data from database")
            slotbook_query = """
            SELECT * FROM slotbook 
            WHERE execution_id = ?
            AND wh_id = ?  
            AND DATE(inventory_snapshot_date) = DATE(?)
            """
            
            slotbook_df = pd.read_sql_query(
                slotbook_query,
                conn,
                params=[execution_id, wh_id, planning_timestamp.strftime('%Y-%m-%d')]
            )
            
            logger.info(f"Retrieved {len(slotbook_df)} slotbook records")
        
        # Write container data CSV
        container_output_path = tf_input_dir / "container_data.csv"
        containers_df.to_csv(container_output_path, index=False)
        logger.debug(f"Wrote container data: {container_output_path}")
        
        # Write slotbook data CSV  
        slotbook_output_path = tf_input_dir / "slotbook_data.csv"
        slotbook_df.to_csv(slotbook_output_path, index=False)
        logger.debug(f"Wrote slotbook data: {slotbook_output_path}")
        
        # Copy tour formation config
        config_output_path = tf_input_dir / "tour_formation_config.yaml"
        shutil.copy2(tour_formation_config_path, config_output_path)
        logger.debug(f"Copied TF config: {config_output_path}")
        
        logger.info(f"TF inputs created successfully: {tf_input_dir}")
        return tf_input_dir
        
    except Exception as e:
        logger.error(f"Failed to create TF inputs: {e}")
        raise


def validate_tf_input_data(tf_input_dir: Path, logger: Optional[logging.Logger] = None) -> bool:
    """
    Simple validation of generated TF input files.
    
    Args:
        tf_input_dir: Directory containing TF input files
        logger: Optional logger instance
        
    Returns:
        True if validation passes, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    required_files = ["container_data.csv", "slotbook_data.csv", "tour_formation_config.yaml"]
    
    for filename in required_files:
        file_path = tf_input_dir / filename
        if not file_path.exists():
            logger.error(f"Missing required file: {filename}")
            return False
            
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                if len(df) == 0:
                    logger.warning(f"Empty CSV file: {filename}")
                logger.debug(f"Validated {filename}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Invalid CSV file {filename}: {e}")
                return False
    
    logger.info("TF input validation passed")
    return True 