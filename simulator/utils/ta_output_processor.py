"""
TA Output Processor Module

Handles processing Tour Allocation (TA) outputs to database tables and cleanup operations.
This module processes TA CSV outputs and stores them in the following database tables:
- ta_pending_tours_by_aisle
- ta_tours_to_release  
- ta_allocation_metadata
- flexsim_inputs

Author: AI Assistant
Date: 2024
"""

import logging
import sqlite3
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime
import yaml
import re


def process_ta_outputs_to_database(
    execution_id: str,
    wh_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    db_path: Path,
    config_path: Path,
    start_time: datetime,
    logger: logging.Logger
) -> None:
    """
    Process TA output CSV files to database tables.
    
    This function reads TA output CSV files and inserts the data into the appropriate
    database tables with proper execution metadata. It also archives released tours
    by marking them in the ready_to_release_tours table.
    
    Processing Steps:
    1. ta_pending_tours_by_aisle.csv → ta_pending_tours_by_aisle table
    2. tours_to_release.csv → ta_tours_to_release table  
    3. ta_allocation_metadata.csv → ta_allocation_metadata table
    4. tours_to_release.csv → flexsim_inputs table (transformed)
    5. Archive released tours → ready_to_release_tours table (archived_flag = 1)
    6. Update container release status → containers table (released_flag = 1, tour_id, release_datetime)
    
    Parameters
    ----------
    execution_id : str
        Unique execution identifier
    wh_id : str
        Warehouse identifier
    planning_datetime : datetime
        Planning timestamp for this TA run
    output_dir : Path
        Directory containing TA output CSV files
    db_path : Path
        Path to SQLite database file
    config_path : Path
        Path to sim_config.yaml file
    start_time : datetime
        Simulation start time for FlexSim time calculations
    logger : logging.Logger
        Logger instance for tracking operations
        
    Raises
    ------
    FileNotFoundError
        If required output files are missing
    sqlite3.Error
        If database operations fail
    """
    logger.info(f"Starting TA output processing to database for execution {execution_id}")
    
    # Convert paths to Path objects if they aren't already
    output_dir = Path(output_dir)
    db_path = Path(db_path)
    
    # Verify output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"TA output directory not found: {output_dir}")
    
    # Connect to database
    with sqlite3.connect(db_path) as conn:
        logger.info(f"Connected to database: {db_path}")
        
        # Process each TA output table
        _process_ta_pending_tours_by_aisle(
            conn, execution_id, wh_id, planning_datetime, output_dir, logger
        )
        
        _process_ta_tours_to_release(
            conn, execution_id, wh_id, planning_datetime, output_dir, logger
        )
        
        _process_ta_allocation_metadata(
            conn, execution_id, wh_id, planning_datetime, output_dir, logger
        )
        
        _process_flexsim_inputs(
            conn, execution_id, wh_id, planning_datetime, output_dir, logger
        )
        
        # Step 5: Archive released tours
        logger.info("Step 5: Archiving released tours")
        _archive_released_tours(conn, execution_id, planning_datetime, output_dir, logger)
        
        # Step 6: Update container release status
        logger.info("Step 6: Updating container release status")
        _update_released_container_status(conn, execution_id, planning_datetime, output_dir, logger)
        
        # Commit all changes
        conn.commit()
        logger.info(f"TA output processing completed successfully for execution {execution_id}")
    
    # Export FlexSim inputs to Excel file
    try:
        export_flexsim_inputs_to_excel(
            execution_id=execution_id,
            planning_datetime=planning_datetime,
            start_time=start_time,
            db_path=db_path,
            config_path=config_path,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to export FlexSim inputs to Excel: {str(e)}")
        # Don't raise - this is supplementary functionality


def _process_ta_pending_tours_by_aisle(
    conn: sqlite3.Connection,
    execution_id: str,
    wh_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Process ta_pending_tours_by_aisle.csv to database."""
    csv_file = output_dir / 'ta_pending_tours_by_aisle.csv'
    
    if not csv_file.exists():
        logger.warning(f"ta_pending_tours_by_aisle.csv not found: {csv_file}")
        return
    
    logger.info(f"Processing ta_pending_tours_by_aisle.csv: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    if df.empty:
        logger.warning("ta_pending_tours_by_aisle.csv is empty")
        return
    
    # Add execution metadata
    df['execution_id'] = execution_id
    df['wh_id'] = wh_id
    df['planning_datetime'] = planning_datetime
    
    # Ensure required columns exist
    required_columns = ['execution_id', 'wh_id', 'planning_datetime', 'aisle', 'tour_count', 'quantity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in ta_pending_tours_by_aisle.csv: {missing_columns}")
    
    # Insert into database
    df[required_columns].to_sql('ta_pending_tours_by_aisle', conn, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} rows into ta_pending_tours_by_aisle table")


def _process_ta_tours_to_release(
    conn: sqlite3.Connection,
    execution_id: str,
    wh_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Process tours_to_release.csv to ta_tours_to_release database table."""
    csv_file = output_dir / 'tours_to_release.csv'
    
    if not csv_file.exists():
        logger.warning(f"tours_to_release.csv not found: {csv_file}")
        return
    
    logger.info(f"Processing tours_to_release.csv: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    if df.empty:
        logger.warning("tours_to_release.csv is empty")
        return
    
    # Add execution metadata
    df['execution_id'] = execution_id
    df['wh_id'] = wh_id
    df['planning_datetime'] = planning_datetime
    
    # Map CSV columns to database columns
    column_mapping = {
        'sku': 'item_number',
        'quantity': 'pick_qty',
        'optimal_pick_location': 'location_id'
    }
    
    # Rename columns to match database schema
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = [
        'execution_id', 'wh_id', 'planning_datetime', 'tour_id', 'container_id', 
        'item_number', 'pick_qty', 'location_id', 'picking_flow_as_int'
    ]
    
    # Add optional columns if they exist
    optional_columns = ['aisle_sequence', 'original_promised_pull_date']
    available_columns = required_columns + [col for col in optional_columns if col in df.columns]
    
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in tours_to_release.csv: {missing_required}")
    
    # Insert into database
    df[available_columns].to_sql('ta_tours_to_release', conn, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} rows into ta_tours_to_release table")


def _process_ta_allocation_metadata(
    conn: sqlite3.Connection,
    execution_id: str,
    wh_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Process ta_allocation_metadata.csv to database."""
    csv_file = output_dir / 'ta_allocation_metadata.csv'
    
    if not csv_file.exists():
        logger.warning(f"ta_allocation_metadata.csv not found: {csv_file}")
        return
    
    logger.info(f"Processing ta_allocation_metadata.csv: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    if df.empty:
        logger.warning("ta_allocation_metadata.csv is empty")
        return
    
    # Add execution metadata
    df['execution_id'] = execution_id
    df['wh_id'] = wh_id
    df['planning_datetime'] = planning_datetime
    
    # Ensure required columns exist
    required_columns = [
        'execution_id', 'wh_id', 'planning_datetime', 'solve_time', 
        'tour_count_target', 'tour_count_released', 'total_aisle_concurrency',
        'maximum_aisle_concurrency', 'total_slack'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in ta_allocation_metadata.csv: {missing_columns}")
    
    # Insert into database
    df[required_columns].to_sql('ta_allocation_metadata', conn, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} rows into ta_allocation_metadata table")


def _process_flexsim_inputs(
    conn: sqlite3.Connection,
    execution_id: str,
    wh_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Process tours_to_release.csv to flexsim_inputs database table."""
    csv_file = output_dir / 'tours_to_release.csv'
    
    if not csv_file.exists():
        logger.warning(f"tours_to_release.csv not found for flexsim_inputs processing: {csv_file}")
        return
    
    logger.info(f"Processing tours_to_release.csv for flexsim_inputs: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    if df.empty:
        logger.warning("tours_to_release.csv is empty for flexsim_inputs")
        return
    
    # Add execution metadata
    df['execution_id'] = execution_id
    df['wh_id'] = wh_id
    df['planning_datetime'] = planning_datetime
    
    # Map CSV columns to database columns for flexsim_inputs
    column_mapping = {
        'sku': 'item_number',
        'quantity': 'pick_qty',
        'optimal_pick_location': 'location_id'
    }
    
    # Rename columns to match database schema
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist for flexsim_inputs
    required_columns = [
        'execution_id', 'wh_id', 'planning_datetime', 'tour_id', 'container_id',
        'item_number', 'pick_qty', 'location_id', 'picking_flow_as_int'
    ]
    
    # Add optional columns if they exist
    optional_columns = ['aisle_sequence', 'original_promised_pull_date']
    available_columns = required_columns + [col for col in optional_columns if col in df.columns]
    
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for flexsim_inputs: {missing_required}")
    
    # Insert into database
    df[available_columns].to_sql('flexsim_inputs', conn, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} rows into flexsim_inputs table")


def _archive_released_tours(
    conn: sqlite3.Connection,
    execution_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Archive tours that were selected for release by marking them in ready_to_release_tours table.
    
    This function reads the tours_to_release.csv file to identify which tours were selected
    by the TA model, then updates the ready_to_release_tours table to mark them as archived.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    execution_id : str
        Unique execution identifier
    planning_datetime : datetime
        Planning timestamp for this TA run
    output_dir : Path
        Directory containing TA output CSV files
    logger : logging.Logger
        Logger instance for tracking operations
    """
    csv_file = output_dir / 'tours_to_release.csv'
    
    if not csv_file.exists():
        logger.warning(f"tours_to_release.csv not found for tour archiving: {csv_file}")
        return
    
    logger.info(f"Archiving released tours from: {csv_file}")
    
    try:
        # Read CSV file to get released tour IDs
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning("tours_to_release.csv is empty - no tours to archive")
            return
        
        # Extract unique tour IDs that were selected for release
        if 'tour_id' not in df.columns:
            logger.error("tour_id column not found in tours_to_release.csv")
            return
        
        released_tour_ids = df['tour_id'].unique().tolist()
        logger.info(f"Found {len(released_tour_ids)} unique tours selected for release")
        
        if not released_tour_ids:
            logger.warning("No tour IDs found to archive")
            return
        
        # Update ready_to_release_tours table to mark selected tours as archived
        placeholders = ','.join(['?' for _ in released_tour_ids])
        archive_query = f"""
            UPDATE ready_to_release_tours 
            SET archived_flag = 1, archived_at_datetime = ?
            WHERE execution_id = ? AND tour_id IN ({placeholders}) AND archived_flag = 0
        """
        
        # Parameters: planning_datetime, execution_id, then all tour_ids
        archive_params = [planning_datetime, execution_id] + released_tour_ids
        
        cursor = conn.execute(archive_query, archive_params)
        rows_updated = cursor.rowcount
        
        logger.info(f"Successfully archived {rows_updated} tours in ready_to_release_tours table")
        
        if rows_updated != len(released_tour_ids*20):
            logger.warning(f"Expected to archive {len(released_tour_ids)} tours but only updated {rows_updated} rows")
            logger.warning("Some tours may have already been archived or not found in ready_to_release_tours table")
        
    except Exception as e:
        logger.error(f"Failed to archive released tours: {str(e)}")
        # Don't raise - this shouldn't break the main TA processing workflow
        logger.warning("Continuing with TA processing despite archiving failure")


def _update_released_container_status(
    conn: sqlite3.Connection,
    execution_id: str,
    planning_datetime: datetime,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Update container release status for containers in released tours.
    
    This function reads the tours_to_release.csv file to identify which tours were selected
    by the TA model, then updates the containers table to mark the associated containers
    as released with the appropriate tour_id and release_datetime.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    execution_id : str
        Unique execution identifier
    planning_datetime : datetime
        Planning timestamp for this TA run
    output_dir : Path
        Directory containing TA output CSV files
    logger : logging.Logger
        Logger instance for tracking operations
    """
    csv_file = output_dir / 'tours_to_release.csv'
    
    if not csv_file.exists():
        logger.warning(f"tours_to_release.csv not found for container updates: {csv_file}")
        return
    
    logger.info(f"Updating container release status from: {csv_file}")
    
    try:
        # Read CSV file to get released tour IDs (same as tour archiving)
        df = pd.read_csv(csv_file)
        
        if df.empty:
            logger.warning("tours_to_release.csv is empty - no containers to update")
            return
        
        if 'tour_id' not in df.columns:
            logger.error("tour_id column not found in tours_to_release.csv")
            return
        
        released_tour_ids = df['tour_id'].unique().tolist()
        logger.info(f"Updating container status for {len(released_tour_ids)} released tours")
        
        if not released_tour_ids:
            logger.warning("No tour IDs found to update container status")
            return
        
        # Get expected container count first
        placeholders = ','.join(['?' for _ in released_tour_ids])
        count_query = f"""
            SELECT COUNT(DISTINCT container_id) as container_count
            FROM ready_to_release_tours 
            WHERE execution_id = ? AND tour_id IN ({placeholders})
            AND archived_flag = 1
        """
        cursor = conn.execute(count_query, [execution_id] + released_tour_ids)
        expected_containers = cursor.fetchone()[0]
        
        if expected_containers == 0:
            logger.warning("No containers found for the specified tour IDs in archived tours")
            return
        
        logger.info(f"Found {expected_containers} containers to update for released tours")
        
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
        
        # Parameters: tour_ids (first IN), planning_datetime, execution_id, execution_id, tour_ids (second IN)
        params = released_tour_ids + [planning_datetime, execution_id, execution_id] + released_tour_ids
        cursor = conn.execute(update_query, params)
        updated_containers = cursor.rowcount
        
        logger.info(f"Successfully updated release status for {updated_containers} containers (expected {expected_containers})")
        
        if updated_containers != expected_containers:
            logger.warning(f"Container update mismatch: updated {updated_containers}, expected {expected_containers}")
            logger.warning("Some containers may have already been released or not found in containers table")
        
    except Exception as e:
        logger.error(f"Failed to update container release status: {str(e)}")
        raise


def cleanup_ta_directories(
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    logger: logging.Logger
) -> None:
    """
    Clean up TA input and output directories after successful database processing.
    
    This function removes temporary directories created during TA processing
    to save disk space and maintain a clean working environment.
    
    Parameters
    ----------
    input_dir : Optional[Path]
        TA input directory to remove (can be None)
    output_dir : Optional[Path]
        TA output directory to remove (can be None)
    logger : logging.Logger
        Logger instance for tracking cleanup operations
        
    Note
    ----
    This function will not raise exceptions if cleanup fails - it will only log warnings.
    This ensures that cleanup failures don't interrupt the main processing workflow.
    """
    logger.info("Starting TA directory cleanup")
    
    directories_cleaned = 0
    directories_failed = 0
    
    # Clean up input directory
    if input_dir and Path(input_dir).exists():
        try:
            shutil.rmtree(input_dir)
            logger.info(f"Successfully removed TA input directory: {input_dir}")
            directories_cleaned += 1
        except Exception as e:
            logger.warning(f"Failed to remove TA input directory {input_dir}: {str(e)}")
            directories_failed += 1
    elif input_dir:
        logger.debug(f"TA input directory does not exist (already cleaned?): {input_dir}")
    
    # Clean up output directory
    if output_dir and Path(output_dir).exists():
        try:
            shutil.rmtree(output_dir)
            logger.info(f"Successfully removed TA output directory: {output_dir}")
            directories_cleaned += 1
        except Exception as e:
            logger.warning(f"Failed to remove TA output directory {output_dir}: {str(e)}")
            directories_failed += 1
    elif output_dir:
        logger.debug(f"TA output directory does not exist (already cleaned?): {output_dir}")
    
    # Summary
    if directories_cleaned > 0:
        logger.info(f"TA cleanup completed: {directories_cleaned} directories removed")
    if directories_failed > 0:
        logger.warning(f"TA cleanup had {directories_failed} failures")
    
    if directories_cleaned == 0 and directories_failed == 0:
        logger.debug("No TA directories found to clean up")


def get_ta_processing_summary(
    execution_id: str,
    db_path: Path,
    logger: logging.Logger
) -> dict:
    """
    Get a summary of TA processing results from the database.
    
    Parameters
    ----------
    execution_id : str
        Execution identifier to summarize
    db_path : Path
        Path to SQLite database file
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Summary statistics of TA processing results
    """
    logger.info(f"Generating TA processing summary for execution {execution_id}")
    
    summary = {
        'execution_id': execution_id,
        'ta_pending_tours_count': 0,
        'ta_tours_released_count': 0,
        'ta_allocation_runs': 0,
        'flexsim_inputs_count': 0,
        'total_tours_processed': 0,
        'total_picks_processed': 0,
        'containers_released_count': 0
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Count pending tours by aisle
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ta_pending_tours_by_aisle WHERE execution_id = ?",
                (execution_id,)
            )
            summary['ta_pending_tours_count'] = cursor.fetchone()[0]
            
            # Count tours released
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT tour_id) FROM ta_tours_to_release WHERE execution_id = ?",
                (execution_id,)
            )
            summary['ta_tours_released_count'] = cursor.fetchone()[0]
            
            # Count allocation runs
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ta_allocation_metadata WHERE execution_id = ?",
                (execution_id,)
            )
            summary['ta_allocation_runs'] = cursor.fetchone()[0]
            
            # Count flexsim inputs
            cursor = conn.execute(
                "SELECT COUNT(*) FROM flexsim_inputs WHERE execution_id = ?",
                (execution_id,)
            )
            summary['flexsim_inputs_count'] = cursor.fetchone()[0]
            
            # Total picks processed
            cursor = conn.execute(
                "SELECT SUM(pick_qty) FROM ta_tours_to_release WHERE execution_id = ?",
                (execution_id,)
            )
            result = cursor.fetchone()[0]
            summary['total_picks_processed'] = result if result else 0
            
            # Count containers released
            cursor = conn.execute(
                "SELECT COUNT(*) FROM containers WHERE execution_id = ? AND released_flag = 1",
                (execution_id,)
            )
            summary['containers_released_count'] = cursor.fetchone()[0]
            
            summary['total_tours_processed'] = summary['ta_tours_released_count']
            
    except sqlite3.Error as e:
        logger.error(f"Database error while generating TA summary: {str(e)}")
        summary['error'] = str(e)
    
    logger.info(f"TA processing summary generated: {summary}")
    return summary


def export_flexsim_inputs_to_excel(
    execution_id: str,
    planning_datetime: datetime,
    start_time: datetime,
    db_path: Path,
    config_path: Path,
    logger: logging.Logger
) -> None:
    """
    Export FlexSim inputs from database to Excel file.
    
    This function queries the flexsim_inputs table, transforms the data according to
    FlexSim requirements, and appends it to the configured Excel file.
    
    Parameters
    ----------
    execution_id : str
        Unique execution identifier
    planning_datetime : datetime
        Planning timestamp for this TA run
    start_time : datetime
        Simulation start time for FlexSim time calculations
    db_path : Path
        Path to SQLite database file
    config_path : Path
        Path to sim_config.yaml file
    logger : logging.Logger
        Logger instance for tracking operations
        
    Raises
    ------
    FileNotFoundError
        If config file is missing
    ValueError
        If required data is missing or invalid
    """
    logger.info(f"Starting FlexSim inputs export for execution {execution_id}, planning_datetime {planning_datetime}")
    
    try:
        # Step 1: Read FlexSim file path from config
        flexsim_file_path = _read_flexsim_config_path(config_path, logger)
        
        # Step 2: Query database for FlexSim inputs
        df = _query_flexsim_inputs_from_database(
            execution_id, planning_datetime, db_path, logger
        )
        
        if df.empty:
            logger.warning(f"No FlexSim inputs found for execution {execution_id}, planning_datetime {planning_datetime}")
            return
        
        # Step 3: Transform data for FlexSim
        df = _transform_data_for_flexsim(df, start_time, planning_datetime, logger)
        
        # Step 4: Handle batch_id uniqueness
        df = _adjust_batch_ids_for_uniqueness(df, flexsim_file_path, logger)
        
        # Step 5: Append to Excel file
        _append_to_flexsim_excel(df, flexsim_file_path, logger)
        
        logger.info(f"Successfully exported {len(df)} FlexSim inputs to {flexsim_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to export FlexSim inputs: {str(e)}")
        raise


def _read_flexsim_config_path(config_path: Path, logger: logging.Logger) -> Path:
    """Read FlexSim inputs file path from configuration."""
    logger.debug(f"Reading FlexSim config path from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        flexsim_path = config['paths']['flexsim_inputs']
        logger.debug(f"FlexSim inputs path from config: {flexsim_path}")
        
        return Path(flexsim_path)
        
    except KeyError as e:
        raise ValueError(f"Missing FlexSim inputs path in config: {e}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to read config file {config_path}: {e}")


def _query_flexsim_inputs_from_database(
    execution_id: str,
    planning_datetime: datetime,
    db_path: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """Query FlexSim inputs from database."""
    logger.debug(f"Querying FlexSim inputs from database for execution {execution_id}")
    
    query = """
    SELECT 
        fi.planning_datetime,
        fi.tour_id,
        fi.container_id,
        fi.item_number,
        fi.pick_qty,
        fi.location_id,
        fi.picking_flow_as_int,
        fi.aisle_sequence,
        fi.original_promised_pull_date
    FROM flexsim_inputs fi
    WHERE fi.execution_id = ? AND fi.planning_datetime = ?
    ORDER BY fi.tour_id, fi.picking_flow_as_int
    """
    
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(execution_id, planning_datetime))
        
        logger.debug(f"Retrieved {len(df)} FlexSim input records from database")
        return df
        
    except sqlite3.Error as e:
        raise ValueError(f"Database query failed: {e}")


def _transform_data_for_flexsim(
    df: pd.DataFrame,
    start_time: datetime,
    planning_datetime: datetime,
    logger: logging.Logger
) -> pd.DataFrame:
    """Transform data according to FlexSim requirements."""
    logger.debug("Transforming data for FlexSim format")
    
    # Calculate flexsim_time in seconds
    df['flexsim_time'] = _calculate_flexsim_time(start_time, planning_datetime)
    
    # Generate batch_id from tour_id
    df['batch_id'] = df['tour_id'].apply(_generate_batch_id_from_tour_id)
    
    # Map column names to FlexSim format
    column_mapping = {
        'container_id': 'containerID',
        'pick_qty': 'tran_qty',
        'original_promised_pull_date': 'original_promised_pull_datetime'
    }
    df = df.rename(columns=column_mapping)
    
    # Select and order columns as required
    required_columns = [
        'flexsim_time', 'batch_id', 'containerID', 'item_number', 'tran_qty',
        'location_id', 'picking_flow_as_int', 'aisle_sequence', 'original_promised_pull_datetime'
    ]
    
    # Handle missing optional columns
    for col in required_columns:
        if col not in df.columns:
            if col == 'aisle_sequence':
                df[col] = 0  # Default value
            elif col == 'original_promised_pull_datetime':
                df[col] = None  # Allow null
            else:
                raise ValueError(f"Required column {col} is missing from data")
    
    df = df[required_columns]
    
    # Sort as required
    df = df.sort_values(by=['batch_id', 'picking_flow_as_int'])
    
    logger.debug(f"Data transformation completed, {len(df)} records ready for export")
    return df


def _calculate_flexsim_time(start_time: datetime, planning_datetime: datetime) -> float:
    """Calculate FlexSim time in seconds."""
    time_diff = start_time - planning_datetime
    return time_diff.total_seconds()


def _generate_batch_id_from_tour_id(tour_id: str) -> int:
    """Convert tour_id to numeric batch_id."""
    # Extract numeric part from tour_id using regex
    match = re.search(r'(\d+)', str(tour_id))
    if match:
        return int(match.group(1))
    else:
        # Fallback: use hash of tour_id
        return abs(hash(str(tour_id))) % 1000000


def _adjust_batch_ids_for_uniqueness(
    df: pd.DataFrame,
    flexsim_file_path: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """Adjust batch_ids to ensure uniqueness across iterations."""
    logger.debug("Adjusting batch_ids for uniqueness")
    
    max_existing_batch_id = 1
    
    try:
        if flexsim_file_path.is_file() and flexsim_file_path.stat().st_size > 0:
            existing_df = pd.read_excel(flexsim_file_path)
            if not existing_df.empty and 'batch_id' in existing_df.columns:
                numeric_batch_ids = pd.to_numeric(existing_df['batch_id'], errors='coerce').dropna()
                if not numeric_batch_ids.empty:
                    max_existing_batch_id = int(numeric_batch_ids.max())
                    logger.info(f"Max existing batch_id from {flexsim_file_path} is {max_existing_batch_id}")
    except Exception as e:
        logger.error(f"Error reading {flexsim_file_path} to determine max batch_id: {e}")
    
    if not df.empty and 'batch_id' in df.columns:
        logger.info(f"Original min batch_id in current run: {df['batch_id'].min()}, max: {df['batch_id'].max()}")
        df['batch_id'] = df['batch_id'].astype(int) + max_existing_batch_id
        logger.info(f"Adjusted min batch_id in current run: {df['batch_id'].min()}, max: {df['batch_id'].max()} (offset by {max_existing_batch_id})")
    
    return df


def _append_to_flexsim_excel(
    df: pd.DataFrame,
    flexsim_file_path: Path,
    logger: logging.Logger
) -> None:
    """Append data to FlexSim Excel file."""
    logger.debug(f"Appending {len(df)} records to FlexSim Excel file: {flexsim_file_path}")
    
    try:
        # Ensure parent directory exists
        flexsim_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has data
        if flexsim_file_path.is_file() and flexsim_file_path.stat().st_size > 0:
            # Append to existing file
            existing_df = pd.read_excel(flexsim_file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            logger.info(f"Appending to existing file with {len(existing_df)} existing records")
        else:
            # Create new file
            combined_df = df
            logger.info("Creating new FlexSim inputs file")
        
        # Write to Excel
        combined_df.to_excel(flexsim_file_path, index=False)
        logger.info(f"Successfully wrote {len(combined_df)} total records to {flexsim_file_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to write FlexSim Excel file {flexsim_file_path}: {e}")