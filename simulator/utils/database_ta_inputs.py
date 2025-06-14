"""
Database-driven Tour Allocation input generation utilities.

This module provides functions to create TA inputs directly from the SQLite database
instead of manual file consolidation, enabling seamless integration with the simulator workflow.
"""

import logging
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil


def create_ta_inputs_from_database(
    execution_id: str,
    planning_timestamp: datetime,
    db_path: Path,
    wh_id: str,
    base_input_dir: Path,
    tour_allocation_config_path: Path,
    data_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create TA inputs from database for a single planning timestamp.
    Generates the traditional 5 CSV files expected by TA model.
    
    Creates directory structure: base_input_dir/{wh_id}/{YYYYMMDD_HHMMSS}/
    Example: input/MDW1/20241127_143000/
    
    Args:
        execution_id: Database execution ID to filter data
        planning_timestamp: Planning timestamp for TA run
        db_path: Path to SQLite database
        wh_id: Warehouse ID from config
        base_input_dir: Base directory for TA inputs (typically "input")
        tour_allocation_config_path: Path to TA config file
        data_dir: Directory containing pending_tours_by_aisle.csv
        logger: Optional logger instance
        
    Returns:
        Path to created TA input directory (e.g., input/MDW1/20241127_143000/)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Creating TA inputs from database for {planning_timestamp}")
    logger.debug(f"Execution ID: {execution_id}, Warehouse: {wh_id}")
    
    # Create directory structure: input/{wh_id}/{planning_timestamp}/
    ta_timestamp_str = planning_timestamp.strftime('%Y%m%d_%H%M%S')
    ta_input_dir = base_input_dir / wh_id / ta_timestamp_str
    ta_input_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Created TA input directory: {ta_input_dir}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Query database for tour data
            logger.debug("Querying tour formation data from database")
            tour_formation_df = _query_tour_formation_details(conn, execution_id, wh_id, logger)
            
            if tour_formation_df.empty:
                logger.warning("No ready-to-release tours found in database")
                # Still create empty files for TA model compatibility
                _create_empty_csv_files(ta_input_dir, logger)
            else:
                logger.info(f"Retrieved {len(tour_formation_df)} tour formation records")
                
                # Calculate aisle ranges dynamically
                logger.debug("Calculating aisle ranges from tour formation data")
                aisle_ranges_df = _calculate_aisle_ranges(conn, execution_id, wh_id, logger)
                
                # Generate CSV dataframes
                logger.debug("Generating TA input CSV dataframes")
                container_assignments_df = _generate_container_assignments(tour_formation_df)
                pick_assignments_df = _generate_pick_assignments(tour_formation_df)
                container_tours_df = _generate_container_tours(tour_formation_df)
                
                # Write CSV files
                csv_files = {
                    'container_assignments': container_assignments_df,
                    'pick_assignments': pick_assignments_df,
                    'aisle_ranges': aisle_ranges_df,
                    'container_tours': container_tours_df
                }
                _write_ta_csv_files(ta_input_dir, csv_files, logger)
        
        # Copy pending tours file from data directory
        _copy_pending_tours_file(data_dir, ta_input_dir, logger)
        
        # Copy tour allocation config
        config_output_path = ta_input_dir / "tour_allocation_config.yaml"
        
        # Resolve config path to absolute path
        tour_allocation_config_path = tour_allocation_config_path.resolve()
        logger.debug(f"Looking for TA config at: {tour_allocation_config_path.absolute()}")
        
        if not tour_allocation_config_path.exists():
            logger.error(f"Tour allocation config not found: {tour_allocation_config_path.absolute()}")
            raise FileNotFoundError(f"Tour allocation config not found: {tour_allocation_config_path}")
        
        shutil.copy2(tour_allocation_config_path, config_output_path)
        logger.debug(f"Copied TA config: {tour_allocation_config_path} -> {config_output_path}")
        
        logger.info(f"TA inputs created successfully: {ta_input_dir}")
        return ta_input_dir
        
    except Exception as e:
        logger.error(f"Failed to create TA inputs: {e}")
        raise


def _query_tour_formation_details(conn, execution_id: str, wh_id: str, logger) -> pd.DataFrame:
    """Query detailed tour formation data for ready-to-release tours."""
    query = """
    SELECT tf.tour_id, tf.container_id, tf.item_number, 
           COALESCE(tf.pick_location, 'UNKNOWN') as pick_location, 
           tf.sequence_order, tf.pick_quantity, tf.cluster_id
    FROM tf_tour_formation tf
    JOIN ready_to_release_tours r ON tf.execution_id = r.execution_id 
                                  AND tf.tour_id = r.tour_id 
                                  AND tf.container_id = r.container_id
    WHERE tf.execution_id = ?
    AND tf.wh_id = ?  
    AND r.archived_flag = 0
    ORDER BY tf.tour_id, tf.container_id, tf.sequence_order
    """
    
    df = pd.read_sql_query(query, conn, params=[execution_id, wh_id])
    logger.debug(f"Tour formation query returned {len(df)} records")
    return df


def _calculate_aisle_ranges(conn, execution_id: str, wh_id: str, logger) -> pd.DataFrame:
    """Calculate aisle ranges dynamically from tour formation data, including total slack."""
    query = """
    SELECT tf.tour_id, 
           MIN(cd.aisle_sequence) as min_aisle,
           MAX(cd.aisle_sequence) as max_aisle,
           COUNT(DISTINCT cd.aisle_sequence) as distinct_aisle_count,
           COALESCE(SUM(cs.slack_minutes), 0.0) as total_slack
    FROM tf_tour_formation tf
    JOIN ready_to_release_tours r ON tf.execution_id = r.execution_id 
                                  AND tf.tour_id = r.tour_id 
                                  AND tf.container_id = r.container_id
    JOIN container_details cd ON tf.execution_id = cd.execution_id
                              AND tf.container_id = cd.container_id 
                              AND tf.item_number = cd.item_number
    LEFT JOIN tf_container_slack cs ON tf.execution_id = cs.execution_id
                                    AND tf.container_id = cs.container_id
                                    AND cs.planning_datetime = r.created_at_datetime
    WHERE tf.execution_id = ?
    AND tf.wh_id = ?
    AND r.archived_flag = 0
    GROUP BY tf.tour_id
    ORDER BY tf.tour_id
    """
    
    df = pd.read_sql_query(query, conn, params=[execution_id, wh_id])
    logger.debug(f"Aisle ranges calculated for {len(df)} tours with total slack")
    logger.debug(f"Total slack range: {df['total_slack'].min():.2f} to {df['total_slack'].max():.2f} minutes")
    return df


def _generate_container_assignments(tour_formation_df: pd.DataFrame) -> pd.DataFrame:
    """Generate container_assignments.csv data from tour formation."""
    if tour_formation_df.empty:
        return pd.DataFrame(columns=['container_id', 'tour_id'])
    
    # Get unique container-tour pairs (note the order: container_id first, then tour_id)
    container_assignments = tour_formation_df[['container_id', 'tour_id']].drop_duplicates()
    return container_assignments.sort_values(['tour_id', 'container_id']).reset_index(drop=True)


def _generate_pick_assignments(tour_formation_df: pd.DataFrame) -> pd.DataFrame:
    """Generate pick_assignments.csv data from tour formation."""
    if tour_formation_df.empty:
        return pd.DataFrame(columns=['container_id', 'sku', 'aisle', 'quantity'])
    
    # Select and rename columns for pick assignments
    pick_assignments = tour_formation_df[[
        'container_id', 'item_number', 'pick_quantity'
    ]].copy()
    
    # Rename columns to match expected structure
    pick_assignments = pick_assignments.rename(columns={
        'item_number': 'sku',
        'pick_quantity': 'quantity'
    })
    
    # Extract aisle number from optimal_pick_location or pick_location
    def extract_aisle_from_location(location):
        """Extract aisle number from location string."""
        if pd.isna(location) or location == '':
            return 1  # Default aisle
        try:
            import re
            location_str = str(location)
            
            # Look for patterns like 'G-23-A02', '01-22-A02', etc.
            # First try to find the first number which is often the aisle
            numbers = re.findall(r'\d+', location_str)
            if numbers:
                first_num = int(numbers[0])
                # If the first number seems reasonable as an aisle (1-50), use it
                if 1 <= first_num <= 50:
                    return first_num
                # Otherwise try the second number if available
                elif len(numbers) > 1:
                    second_num = int(numbers[1])
                    if 1 <= second_num <= 50:
                        return second_num
                    
            # If no good numbers, try to extract letter and convert to number (A=1, B=2, etc.)
            letters = re.findall(r'^[A-Z]', location_str)
            if letters:
                return ord(letters[0]) - ord('A') + 1
                
            return 1  # Fallback to aisle 1
        except:
            return 1  # Fallback to aisle 1
    
    # Check if we have location data to extract aisle from
    if 'optimal_pick_location' in tour_formation_df.columns:
        pick_assignments['aisle'] = tour_formation_df['optimal_pick_location'].apply(extract_aisle_from_location)
    elif 'pick_location' in tour_formation_df.columns:
        pick_assignments['aisle'] = tour_formation_df['pick_location'].apply(extract_aisle_from_location)
    else:
        # Fallback: assign default aisle
        pick_assignments['aisle'] = 1
    
    return pick_assignments.sort_values(['container_id', 'aisle']).reset_index(drop=True)


def _generate_container_tours(tour_formation_df: pd.DataFrame) -> pd.DataFrame:
    """Generate container_tours.csv data from tour formation."""
    if tour_formation_df.empty:
        return pd.DataFrame(columns=['container_id', 'sku', 'quantity', 'tour_id', 'optimal_pick_location', 'picking_flow_as_int'])
    
    # Select all relevant columns for container tours
    base_columns = ['container_id', 'item_number', 'pick_quantity', 'tour_id']
    
    # Add optional columns if they exist
    optional_columns = []
    if 'optimal_pick_location' in tour_formation_df.columns:
        optional_columns.append('optimal_pick_location')
    elif 'pick_location' in tour_formation_df.columns:
        optional_columns.append('pick_location')
        
    if 'picking_flow_as_int' in tour_formation_df.columns:
        optional_columns.append('picking_flow_as_int')
    
    # Select available columns
    available_columns = base_columns + [col for col in optional_columns if col in tour_formation_df.columns]
    container_tours = tour_formation_df[available_columns].copy()
    
    # Rename columns to match expected structure
    container_tours = container_tours.rename(columns={
        'item_number': 'sku',
        'pick_quantity': 'quantity'
    })
    
    # If we used pick_location instead of optimal_pick_location, rename it
    if 'pick_location' in container_tours.columns and 'optimal_pick_location' not in container_tours.columns:
        container_tours = container_tours.rename(columns={'pick_location': 'optimal_pick_location'})
    
    # Add missing columns with default values if they don't exist
    if 'optimal_pick_location' not in container_tours.columns:
        container_tours['optimal_pick_location'] = ''
        
    if 'picking_flow_as_int' not in container_tours.columns:
        container_tours['picking_flow_as_int'] = 0
    
    return container_tours.sort_values(['tour_id', 'container_id']).reset_index(drop=True)


def _write_ta_csv_files(ta_input_dir: Path, dataframes: dict, logger) -> None:
    """Write all TA CSV files to input directory."""
    for filename, df in dataframes.items():
        output_path = ta_input_dir / f"{filename}.csv"
        df.to_csv(output_path, index=False)
        logger.debug(f"Wrote {filename}.csv: {len(df)} rows")


def _copy_pending_tours_file(data_dir: Path, ta_input_dir: Path, logger) -> None:
    """Copy pending_tours_by_aisle.csv from data directory."""
    # Resolve the data_dir to absolute path to handle relative path issues
    data_dir = data_dir.resolve()
    source_path = data_dir / "pending_tours_by_aisle.csv"
    dest_path = ta_input_dir / "pending_tours_by_aisle.csv"
    
    logger.debug(f"Looking for pending tours file at: {source_path.absolute()}")
    
    if not source_path.exists():
        logger.warning(f"Pending tours file not found: {source_path.absolute()}. Creating empty file.")
        # Create empty pending tours file with index to match sample structure
        empty_df = pd.DataFrame(columns=['aisle', 'tour_count', 'quantity'])
        empty_df.to_csv(dest_path, index=True)  # Include index to match sample
        logger.debug(f"Created empty pending_tours_by_aisle.csv: {dest_path}")
    else:
        shutil.copy2(source_path, dest_path)
        logger.debug(f"Copied pending_tours_by_aisle.csv: {source_path} -> {dest_path}")


def _create_empty_csv_files(ta_input_dir: Path, logger) -> None:
    """Create empty CSV files when no tour data is available."""
    empty_files = {
        'container_assignments': pd.DataFrame(columns=['container_id', 'tour_id']),
        'pick_assignments': pd.DataFrame(columns=['container_id', 'sku', 'aisle', 'quantity']),
        'aisle_ranges': pd.DataFrame(columns=['tour_id', 'min_aisle', 'max_aisle', 'total_slack']),
        'container_tours': pd.DataFrame(columns=['container_id', 'sku', 'quantity', 'tour_id', 'optimal_pick_location', 'picking_flow_as_int']),
        'pending_tours_by_aisle': pd.DataFrame(columns=['aisle', 'tour_count', 'quantity'])
    }
    
    for filename, df in empty_files.items():
        output_path = ta_input_dir / f"{filename}.csv"
        # Special handling for pending_tours_by_aisle to match sample structure
        if filename == 'pending_tours_by_aisle':
            df.to_csv(output_path, index=True)
        else:
            df.to_csv(output_path, index=False)
        logger.debug(f"Created empty {filename}.csv")


def validate_ta_input_data(ta_input_dir: Path, logger: Optional[logging.Logger] = None) -> bool:
    """
    Simple validation of generated TA input files.
    
    Args:
        ta_input_dir: Directory containing TA input files
        logger: Optional logger instance
        
    Returns:
        True if validation passes, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    required_files = [
        "container_assignments.csv", 
        "pick_assignments.csv", 
        "aisle_ranges.csv",
        "container_tours.csv",
        "pending_tours_by_aisle.csv", 
        "tour_allocation_config.yaml"
    ]
    
    for filename in required_files:
        file_path = ta_input_dir / filename
        if not file_path.exists():
            logger.error(f"Missing required file: {filename}")
            return False
            
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                if filename == "pending_tours_by_aisle.csv" and len(df) == 0:
                    logger.info(f"Empty CSV file (expected): {filename}")
                else:
                    logger.debug(f"Validated {filename}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Invalid CSV file {filename}: {e}")
                return False
    
    logger.info("TA input validation passed")
    return True


def get_ready_tours_summary(
    execution_id: str,
    db_path: Path,
    wh_id: str,
    logger: Optional[logging.Logger] = None
) -> dict:
    """
    Get summary of ready-to-release tours for diagnostic purposes.
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        logger: Optional logger instance
        
    Returns:
        Dictionary with tour summary statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Count ready tours
            count_query = """
            SELECT COUNT(DISTINCT tour_id) as tour_count,
                   COUNT(*) as container_count,
                   MIN(created_at_datetime) as earliest_creation,
                   MAX(created_at_datetime) as latest_creation
            FROM ready_to_release_tours
            WHERE execution_id = ? 
            AND wh_id = ?
            AND archived_flag = 0
            """
            
            summary_df = pd.read_sql_query(count_query, conn, params=[execution_id, wh_id])
            
            if summary_df.iloc[0]['tour_count'] == 0:
                return {
                    'tour_count': 0,
                    'container_count': 0,
                    'status': 'No ready tours available'
                }
            
            return {
                'tour_count': int(summary_df.iloc[0]['tour_count']),
                'container_count': int(summary_df.iloc[0]['container_count']),
                'earliest_creation': summary_df.iloc[0]['earliest_creation'],
                'latest_creation': summary_df.iloc[0]['latest_creation'],
                'status': 'Ready tours available'
            }
            
    except Exception as e:
        logger.error(f"Failed to get ready tours summary: {e}")
        return {'status': f'Error: {e}'} 