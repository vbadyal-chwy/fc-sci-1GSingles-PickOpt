"""
Simple TF Output Processor for Database Integration.

This module provides functions to process TF output CSV files and store them
in the database using SimulationDBManager.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Import the database manager
import sys
sys.path.append(str(Path(__file__).parent.parent / "data_store" / "core"))
from db_manager import SimulationDBManager


def process_tf_outputs_to_database(
    output_dir: Path,
    db_manager: SimulationDBManager,
    wh_id: str,
    planning_datetime: datetime,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Process all TF output CSV files and store them in database.
    
    This function:
    1. Reads tf_container_slack.csv and stores in tf_container_slack table
    2. Reads tf_container_target.csv and stores in tf_container_target table  
    3. Reads tf_container_clustering.csv and stores in tf_container_clustering table
    4. Reads tf_clustering_metadata.csv and stores in tf_clustering_metadata table
    5. Reads all cluster_{id}_tour_formation.csv files and combines them
    6. Reads all cluster_{id}_solve_metrics.csv files and updates metadata
    7. Creates ready_to_release_tours entries
    
    Parameters
    ----------
    output_dir : Path
        Directory containing TF output CSV files
    db_manager : SimulationDBManager
        Database manager instance
    wh_id : str
        Warehouse ID
    planning_datetime : datetime
        Planning timestamp
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    bool
        True if successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing TF outputs from: {output_dir}")
    
    try:
        # 1. Process container slack calculations (tf_container_slack.csv)
        slack_file = output_dir / "tf_container_slack.csv"
        if slack_file.exists():
            try:
                slack_df = pd.read_csv(slack_file)
                logger.info(f"Found container slack data with {len(slack_df)} records")
                
                success = db_manager.store_tf_slack_calculations(slack_df, planning_datetime)
                if not success:
                    logger.error("Failed to store container slack calculations")
                    return False
            except Exception as e:
                logger.error(f"Failed to process container slack file: {e}")
                return False
        else:
            logger.warning("No tf_container_slack.csv file found")
        
        # 2. Process container target calculations (tf_container_target.csv)
        target_file = output_dir / "tf_container_target.csv"
        if target_file.exists():
            try:
                target_df = pd.read_csv(target_file)
                logger.info(f"Found container target data with {len(target_df)} records")
                
                success = db_manager.store_tf_target_calculations(target_df, planning_datetime)
                if not success:
                    logger.error("Failed to store container target calculations")
                    return False
            except Exception as e:
                logger.error(f"Failed to process container target file: {e}")
                return False
        else:
            logger.warning("No tf_container_target.csv file found")
        
        # 3. Process container clustering assignments (tf_container_clustering.csv)
        clustering_file = output_dir / "tf_container_clustering.csv"
        if clustering_file.exists():
            try:
                clustering_df = pd.read_csv(clustering_file)
                logger.info(f"Found container clustering data with {len(clustering_df)} records")
                
                # Store clustering assignments (metadata will be stored separately)
                success = db_manager.store_tf_clustering_results(
                    clustering_df=clustering_df,
                    metadata_df=pd.DataFrame(),  # Empty - will be processed separately
                    planning_datetime=planning_datetime
                )
                if not success:
                    logger.error("Failed to store container clustering assignments")
                    return False
            except Exception as e:
                logger.error(f"Failed to process container clustering file: {e}")
                return False
        else:
            logger.warning("No tf_container_clustering.csv file found")

        # 4. Process clustering metadata (tf_clustering_metadata.csv)
        metadata_file = output_dir / "tf_clustering_metadata.csv"
        if metadata_file.exists():
            try:
                metadata_df = pd.read_csv(metadata_file)
                logger.info(f"Found clustering metadata with {len(metadata_df)} clusters")
                
                # Store clustering metadata
                success = db_manager.store_tf_clustering_results(
                    clustering_df=pd.DataFrame(),  # Empty - already stored above
                    metadata_df=metadata_df,
                    planning_datetime=planning_datetime
                )
                if not success:
                    logger.error("Failed to store clustering metadata")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to process clustering metadata: {e}")
                return False
        else:
            logger.warning("No clustering metadata file found")
        
        # 5. Process tour formation files (cluster_{id}_tour_formation.csv)
        tour_formation_files = list(output_dir.glob("cluster_*_tour_formation.csv"))
        if tour_formation_files:
            logger.info(f"Found {len(tour_formation_files)} tour formation files")
            
            # Combine all tour formation data
            all_tours = []
            for tf_file in tour_formation_files:
                try:
                    df = pd.read_csv(tf_file)
                    if not df.empty:
                        all_tours.append(df)
                        logger.debug(f"Loaded {len(df)} tours from {tf_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to read {tf_file.name}: {e}")
            
            if all_tours:
                combined_tours_df = pd.concat(all_tours, ignore_index=True)
                logger.info(f"Combined {len(combined_tours_df)} total tour formation records")
                
                # Diagnostic information
                null_pick_locations = combined_tours_df['pick_location'].isnull().sum()
                if null_pick_locations > 0:
                    logger.warning(f"Found {null_pick_locations} records with NULL pick_location values")
                
                # Store in database
                success = db_manager.store_tf_tour_formation(combined_tours_df, planning_datetime)
                if not success:
                    logger.error("Failed to store tour formation data")
                    return False
                
                # Create ready-to-release tours
                ready_tours_df = combined_tours_df[['wh_id','tour_id', 'container_id']].drop_duplicates()
                ready_tours_df['created_at_datetime'] = planning_datetime
                
                success = db_manager.add_tours_to_ready_pool(ready_tours_df, planning_datetime)
                if not success:
                    logger.error("Failed to add tours to ready pool")
                    return False
                    
                logger.info(f"Added {len(ready_tours_df)} tours to ready pool")
            else:
                logger.warning("No tour formation data to process")
        else:
            logger.warning("No tour formation files found")
        
        # 6. Process solve metrics files (cluster_{id}_solve_metrics.csv) and update metadata
        solve_metrics_files = list(output_dir.glob("cluster_*_solve_metrics.csv"))
        if solve_metrics_files:
            logger.info(f"Found {len(solve_metrics_files)} solve metrics files")
            
            # Combine all solve metrics
            all_metrics = []
            for metrics_file in solve_metrics_files:
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        all_metrics.append(df)
                        logger.debug(f"Loaded solve metrics from {metrics_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to read {metrics_file.name}: {e}")
            
            if all_metrics:
                combined_metrics_df = pd.concat(all_metrics, ignore_index=True)
                logger.info(f"Combined {len(combined_metrics_df)} solve metrics records")
                
                # Update clustering metadata with solve metrics
                success = db_manager.update_tf_clustering_metadata_with_solve_metrics(
                    combined_metrics_df, planning_datetime
                )
                if not success:
                    logger.error("Failed to update clustering metadata with solve metrics")
                    return False
            else:
                logger.warning("No solve metrics data to process")
        else:
            logger.warning("No solve metrics files found")
        
        logger.info("TF output processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"TF output processing failed: {e}")
        return False


def cleanup_tf_directories(
    base_input_dir: Path,
    base_output_dir: Path, 
    base_working_dir: Path,
    wh_id: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Clean up TF directories for specific warehouse after successful processing.
    
    Parameters
    ----------
    base_input_dir : Path
        Base input directory (pick_optimization/input)
    base_output_dir : Path
        Base output directory (pick_optimization/output)
    base_working_dir : Path
        Base working directory (pick_optimization/working)
    wh_id : str
        Warehouse ID to clean up (only this subdirectory will be cleaned)
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    bool
        True if successful, False if any failures (non-blocking)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TF directory cleanup for warehouse: {wh_id}")
    
    cleanup_success = True
    directories_to_clean = [
        (base_input_dir / wh_id, "input"),
        (base_output_dir / wh_id, "output"), 
        (base_working_dir / wh_id, "working")
    ]
    
    for dir_path, dir_type in directories_to_clean:
        try:
            if dir_path.exists():
                # Count files before deletion
                files_before = list(dir_path.rglob("*"))
                file_count = len([f for f in files_before if f.is_file()])
                
                if file_count > 0:
                    # Remove all contents but keep the directory
                    import shutil
                    for item in dir_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    
                    logger.info(f"Cleaned {file_count} files from {dir_type} directory: {dir_path}")
                else:
                    logger.info(f"No files to clean in {dir_type} directory: {dir_path}")
            else:
                logger.info(f"Directory does not exist: {dir_path}")
                
        except Exception as e:
            logger.warning(f"Failed to clean {dir_type} directory {dir_path}: {e}")
            cleanup_success = False
    
    if cleanup_success:
        logger.info(f"TF directory cleanup completed successfully for {wh_id}")
    else:
        logger.warning(f"TF directory cleanup completed with some failures for {wh_id}")
    
    return cleanup_success


def get_tf_output_summary(output_dir: Path, logger: Optional[logging.Logger] = None) -> dict:
    """
    Get summary of TF output files in directory.
    
    Parameters
    ----------
    output_dir : Path
        Output directory to analyze
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    dict
        Summary of files found
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    summary = {
        'container_slack_file_exists': False,
        'container_target_file_exists': False,
        'container_clustering_file_exists': False,
        'metadata_file_exists': False,
        'tour_formation_files': [],
        'solve_metrics_files': [],
        'total_files': 0
    }
    
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return summary
    
    # Check for container slack file
    slack_file = output_dir / "tf_container_slack.csv"
    summary['container_slack_file_exists'] = slack_file.exists()
    
    # Check for container target file
    target_file = output_dir / "tf_container_target.csv"
    summary['container_target_file_exists'] = target_file.exists()
    
    # Check for container clustering file
    clustering_file = output_dir / "tf_container_clustering.csv"
    summary['container_clustering_file_exists'] = clustering_file.exists()
    
    # Check for metadata file
    metadata_file = output_dir / "tf_clustering_metadata.csv"
    summary['metadata_file_exists'] = metadata_file.exists()
    
    # Check for tour formation files
    tour_formation_files = list(output_dir.glob("cluster_*_tour_formation.csv"))
    summary['tour_formation_files'] = [f.name for f in tour_formation_files]
    
    # Check for solve metrics files  
    solve_metrics_files = list(output_dir.glob("cluster_*_solve_metrics.csv"))
    summary['solve_metrics_files'] = [f.name for f in solve_metrics_files]
    
    # Count all CSV files
    all_csv_files = list(output_dir.glob("*.csv"))
    summary['total_files'] = len(all_csv_files)
    summary['all_csv_files'] = [f.name for f in all_csv_files]
    
    logger.info(f"TF output summary: "
                f"slack: {summary['container_slack_file_exists']}, "
                f"target: {summary['container_target_file_exists']}, "
                f"clustering: {summary['container_clustering_file_exists']}, "
                f"metadata: {summary['metadata_file_exists']}, "
                f"{len(tour_formation_files)} tour formation, "
                f"{len(solve_metrics_files)} solve metrics")
    
    return summary 