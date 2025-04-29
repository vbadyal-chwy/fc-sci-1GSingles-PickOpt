"""
Utility functions external to the model logic (for Platform)
"""

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List
import shutil

# --- Utility Function - Ensure Directory Exists ---
def ensure_dir_exists(dir_path: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.
    """
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Utility Function - Get Project Root ---
def get_project_root() -> Path:
    """Returns the root directory of the project based on this file's location."""

    return Path(__file__).parent.parent

# --- Function to Create Tour Formation Inputs ---
def create_tour_formation_inputs(
    fc_id: str,
    tf_planning_timestamp: datetime,
    base_input_dir: Path,
    container_data_path: Path,
    slotbook_data_path: Path,
    tour_formation_config_path: Path,
    logger: logging.Logger = None
) -> Path:
    """
    Creates the necessary input files and directory structure for a Tour Formation run.

    Filters container data based on the planning timestamp and copies
    the filtered container data, slotbook data, and tour formation config
    to the FC-specific, timestamp-specific input directory.

    Args:
        fc_id: Facility Center ID.
        tf_planning_timestamp: Timestamp used for the Tour Formation run.
        base_input_dir: The base directory where model inputs are stored.
        container_data_path: Path to the source container_data.csv file.
        slotbook_data_path: Path to the source slotbook_data.csv file.
        tour_formation_config_path: Path to the source tour_formation_config.yaml file.
        logger: Optional logger instance. If None, a default logger is set up.

    Returns:
        The path to the created Tour Formation input directory.

    Raises:
        FileNotFoundError: If container_data, slotbook_data, or tour_formation_config files are not found.
        Exception: For other potential errors during file processing or copying.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Starting creation of TF inputs.")
    logger.info(f"FC ID: {fc_id}")
    logger.info(f"TF Timestamp: {tf_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Base Input Dir: {base_input_dir}")
    logger.debug(f"Container Data Source: {container_data_path}")
    logger.debug(f"Slotbook Data Source: {slotbook_data_path}")
    logger.debug(f"Tour Formation Config Source: {tour_formation_config_path}")

    try:
        # Construct FC-specific & Planning Timestamp specific path for TF input
        tf_timestamp_str = tf_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        tf_input_dir = base_input_dir / fc_id / tf_timestamp_str

        logger.debug(f"Target TF Input Directory: {tf_input_dir}")

        # Ensure the TF input directory exists
        ensure_dir_exists(tf_input_dir)
        logger.debug(f"Ensured TF input directory exists: {tf_input_dir}")

        # --- Process Container Data ---
        logger.debug(f"Attempting to read container data: {container_data_path}")
        if not container_data_path.is_file():
            logger.error(f"Container data file not found: {container_data_path}")
            raise FileNotFoundError(f"Container data file not found: {container_data_path}")

        containers_df = pd.read_csv(container_data_path)
        logger.debug(f"Read {len(containers_df)} rows from {container_data_path}")

        # Convert arrive_datetime to datetime objects if not already
        if 'arrive_datetime' in containers_df.columns \
        and not pd.api.types.is_datetime64_any_dtype(containers_df['arrive_datetime']):
             try:
                 containers_df['arrive_datetime'] = pd.to_datetime(containers_df['arrive_datetime'])
                 logger.debug("Converted 'arrive_datetime' column to datetime objects.")
             except Exception as e:
                 logger.error(f"Error converting 'arrive_datetime' to datetime: {e}", exc_info=True)
                 raise ValueError("Failed to convert 'arrive_datetime' column to datetime objects.")

        # Filter container data
        logger.info(f"Filtering container data based on timestamp: {tf_planning_timestamp}")
        # Ensure required columns exist before filtering
        required_cols = ['arrive_datetime', 'released_flag']
        if not all(col in containers_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in containers_df.columns]
            logger.error(f"Missing required columns in container data: {missing_cols}")
            raise ValueError(f"Container data is missing required columns: {missing_cols}")


        filtered_containers_df = containers_df[
            (containers_df['arrive_datetime'] <= tf_planning_timestamp) &
            (~containers_df['released_flag'].astype(bool)) # Ensure released_flag is treated as boolean
        ].copy()
        logger.info(f"Filtered container data: {len(filtered_containers_df)} rows remaining.")

        # Write filtered container data to TF input directory
        filtered_container_output_path = tf_input_dir / "container_data.csv"
        filtered_containers_df.to_csv(filtered_container_output_path, index=False)
        logger.debug(f"Successfully wrote filtered container data: {filtered_container_output_path}")


        # --- Filter and Copy Slotbook Data ---
        logger.debug(f"Attempting to read slotbook data: {slotbook_data_path}")
        if not slotbook_data_path.is_file():
            logger.error(f"Slotbook data file not found: {slotbook_data_path}")
            raise FileNotFoundError(f"Slotbook data file not found: {slotbook_data_path}")

        # Read slotbook data
        slotbook_df = pd.read_csv(slotbook_data_path)
        logger.debug(f"Read {len(slotbook_df)} rows from {slotbook_data_path}")
        
        # Extract just the date part from the planning timestamp
        planning_date = tf_planning_timestamp.date()
        logger.info(f"Filtering slotbook data for inventory_snapshot_date == {planning_date}")
        
        # Ensure inventory_snapshot_date column exists
        if 'inventory_snapshot_date' not in slotbook_df.columns:
            logger.error(f"'inventory_snapshot_date' column not found in {slotbook_data_path}")
            raise ValueError("Slotbook data is missing required column: inventory_snapshot_date")
        
        # Convert inventory_snapshot_date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(slotbook_df['inventory_snapshot_date']):
            try:
                slotbook_df['inventory_snapshot_date'] = pd.to_datetime(slotbook_df['inventory_snapshot_date'])
                logger.debug("Converted 'inventory_snapshot_date' column to datetime objects.")
            except Exception as e:
                logger.error(f"Error converting 'inventory_snapshot_date' to datetime: {e}", exc_info=True)
                raise ValueError("Failed to convert 'inventory_snapshot_date' column to datetime objects.")
        
        # Filter slotbook data to only include rows matching the planning date
        filtered_slotbook_df = slotbook_df[slotbook_df['inventory_snapshot_date'].dt.date == planning_date].copy()
        logger.info(f"Filtered slotbook data: {len(filtered_slotbook_df)} rows remaining.")
        
        if filtered_slotbook_df.empty:
            logger.warning(f"No slotbook data found for date {planning_date}. The filtered dataframe is empty.")
        
        # Write filtered slotbook data to output
        slotbook_output_path = tf_input_dir / "slotbook_data.csv"
        filtered_slotbook_df.to_csv(slotbook_output_path, index=False)
        logger.debug(f"Successfully wrote filtered slotbook data to: {slotbook_output_path}")

        # --- Copy Tour Formation Config ---
        logger.debug(f"Attempting to copy tour formation config: {tour_formation_config_path}")
        if not tour_formation_config_path.is_file():
            logger.error(f"Tour formation config file not found: {tour_formation_config_path}")
            raise FileNotFoundError(f"Tour formation config file not found: {tour_formation_config_path}")

        config_output_path = tf_input_dir / "tour_formation_config.yaml"
        shutil.copy2(tour_formation_config_path, config_output_path)
        logger.debug(f"Successfully copied tour formation config to: {config_output_path}")

        logger.info("Creation of TF inputs completed successfully.")
        return tf_input_dir

    except FileNotFoundError as fnf_error:
        logger.error(f"Input file not found: {fnf_error}")
        raise # Re-raise the specific error
    except ValueError as val_error:
         logger.error(f"Data validation error: {val_error}")
         raise # Re-raise the specific error
    except Exception as e:
        logger.error(f"An unexpected error occurred during TF input creation: {e}", exc_info=True)
        raise # Re-raise the general exception

# --- Function to Create Ready to Release Tours ---
def create_ready_to_release_tours(
    fc_id: str,
    tf_planning_timestamp: datetime,
    base_tf_output_dir: Path,
    data_dir: Path,
    logger: logging.Logger = None
) -> None:
    """
    Consolidates Tour Formation (TF) output files for multiple clusters into single files
    for ready-to-release tours.

    Args:
        fc_id: Facility Center ID.
        tf_planning_timestamp: Timestamp used for the Tour Formation run.
        base_tf_output_dir: Base directory containing TF outputs.
        data_dir: Directory where consolidated ready-to-release tour files will be saved.
        logger: Optional logger instance. If None, a default logger is set up.

    Raises:
        FileNotFoundError: If the metadata file or cluster files are not found.
        ValueError: If required columns are missing in input files.
        Exception: For other potential errors during file processing or copying.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Starting consolidation of TF outputs for ready-to-release tours.")
    logger.info(f"FC ID: {fc_id}")
    logger.info(f"TF Timestamp: {tf_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Base TF Output Dir: {base_tf_output_dir}")
    logger.debug(f"Data Dir: {data_dir}")

    try:
        # Construct FC-specific & Planning Timestamp specific path for TF output
        tf_timestamp_str = tf_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        tf_output_dir = base_tf_output_dir / fc_id / tf_timestamp_str

        logger.debug(f"Target TF Output Directory: {tf_output_dir}")

        # Ensure the data directory exists
        ensure_dir_exists(data_dir)
        logger.debug(f"Ensured data directory exists: {data_dir}")

        # --- Read Clustering Metadata ---
        metadata_path = tf_output_dir / "clustering_metadata.csv"
        logger.debug(f"Attempting to read metadata file: {metadata_path}")
        try:
            metadata_df = pd.read_csv(metadata_path)
            if 'cluster_id' not in metadata_df.columns:
                logger.error(f"'cluster_id' column not found in {metadata_path}. Cannot proceed.")
                raise ValueError("'cluster_id' column missing in metadata file.")
            cluster_ids = metadata_df['cluster_id'].unique().tolist()
            logger.info(f"Found {len(cluster_ids)} cluster IDs: {cluster_ids}")
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {metadata_path}. Cannot proceed.")
            raise
        except Exception as e:
            logger.error(f"Error reading metadata file {metadata_path}: {e}", exc_info=True)
            raise

        # --- Define File Suffixes Needed for Ready-to-Release Tours ---
        file_categories = {
            "_aisle_ranges.csv": "aisle_ranges.csv",
            "_container_assignments.csv": "container_assignments.csv",
            "_container_tours.csv": "container_tours.csv",
            "_pick_assignments.csv": "pick_assignments.csv"
        }

        # --- Process Each File Category ---
        for suffix, output_filename in file_categories.items():
            logger.info(f"Processing files with suffix: '{suffix}'")
            dfs_to_concat: List[pd.DataFrame] = []

            for cluster_id in cluster_ids:
                input_filename = f"cluster_{cluster_id}{suffix}"
                input_filepath = tf_output_dir / input_filename
                logger.debug(f"Looking for file: {input_filepath}")

                try:
                    # Check if file exists
                    if input_filepath.is_file():
                        df = pd.read_csv(input_filepath)
                        dfs_to_concat.append(df)
                        logger.debug(f"Successfully read and added: {input_filepath}")
                    else:
                        logger.warning(f"File not found for cluster {cluster_id}: {input_filepath}. Skipping.")
                except Exception as e:
                    logger.error(f"Error reading file {input_filepath}: {e}", exc_info=True)

            # --- Concatenate and Write Output ---
            if dfs_to_concat:
                logger.debug(f"Concatenating {len(dfs_to_concat)} DataFrames for '{output_filename}'")
                consolidated_df = pd.concat(dfs_to_concat, ignore_index=True)
                output_filepath = data_dir / output_filename

                try:
                    consolidated_df.to_csv(output_filepath, index=False)
                    logger.debug(f"Successfully wrote consolidated file: {output_filepath}")
                except Exception as e:
                    logger.error(f"Error writing consolidated file {output_filepath}: {e}", exc_info=True)
            else:
                logger.warning(f"No files found for suffix '{suffix}'. No output file '{output_filename}' created.")

        logger.info("Consolidation of TF outputs for ready-to-release tours completed successfully.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Required file not found during consolidation: {fnf_error}")
        raise
    except ValueError as ve:
        logger.error(f"Data validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during consolidation: {e}", exc_info=True)
        raise


# --- Function to Create Tour Allocation Inputs ---
def create_tour_allocation_inputs(
    fc_id: str,
    ta_planning_timestamp: datetime,
    base_ta_input_dir: Path,
    tour_allocation_config_path: Path,
    data_dir: Path,
    logger: logging.Logger = None
) -> None:
    """
    Prepares the input directory for Tour Allocation (TA) using pre-consolidated ready-to-release tour files.

    Args:
        fc_id: Facility Center ID.
        ta_planning_timestamp: Timestamp used for the Tour Allocation run (determines output dir).
        base_ta_input_dir: Base directory where TA inputs should be written.
        tour_allocation_config_path: Path to the source tour_allocation_config.yaml file.
        data_dir: Directory containing the consolidated ready-to-release tour files.
        logger: Optional logger instance. If None, a default logger is set up.

    Raises:
        FileNotFoundError: If the tour_allocation_config file or any consolidated file is not found.
        Exception: For other potential errors during file processing or copying.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Starting preparation of TA inputs.")
    logger.info(f"FC ID: {fc_id}")
    logger.info(f"TA Timestamp: {ta_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Base TA Input Dir: {base_ta_input_dir}")
    logger.debug(f"Tour Allocation Config Source: {tour_allocation_config_path}")
    logger.debug(f"Data Dir: {data_dir}")

    try:
        # Construct FC-specific & Planning Timestamp specific path for TA input
        ta_timestamp_str = ta_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        ta_input_dir = base_ta_input_dir / fc_id / ta_timestamp_str

        logger.debug(f"Target TA Input Directory: {ta_input_dir}")

        # Ensure the TA input directory exists
        ensure_dir_exists(ta_input_dir)
        logger.debug(f"Ensured TA input directory exists: {ta_input_dir}")

        # --- Copy Tour Allocation Config ---
        logger.debug(f"Attempting to copy tour allocation config: {tour_allocation_config_path}")
        if not tour_allocation_config_path.is_file():
            logger.error(f"Tour allocation config file not found: {tour_allocation_config_path}")
            raise FileNotFoundError(f"Tour allocation config file not found: {tour_allocation_config_path}")

        config_output_path = ta_input_dir / "tour_allocation_config.yaml"
        shutil.copy2(tour_allocation_config_path, config_output_path)
        logger.debug(f"Successfully copied tour allocation config to: {config_output_path}")

        # --- Copy Consolidated Ready-to-Release Tour Files ---
        file_categories = [
            "aisle_ranges.csv",
            "container_assignments.csv",
            "container_tours.csv",
            "pick_assignments.csv"
            "pending_tours_by_aisle.csv"
        ]

        for filename in file_categories:
            source_path = data_dir / filename
            if not source_path.is_file():
                logger.warning(f"Consolidated file not found: {source_path}. Skipping.")
                continue
            dest_path = ta_input_dir / filename
            shutil.copy2(source_path, dest_path)
            logger.debug(f"Successfully copied {filename} to: {dest_path}")

        logger.info("Preparation of TA inputs completed successfully.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Required file not found during TA input preparation: {fnf_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during TA input preparation: {e}", exc_info=True)
        raise

# --- Function to Update Container Release Status ---
def update_container_release_status(
    fc_id: str,
    ta_planning_timestamp: datetime,
    base_ta_output_dir: Path,
    container_data_path: Path,
    logger: logging.Logger = None
) -> None:
    """
    Updates the release status of containers based on Tour Allocation output.

    Reads 'tours_to_release.csv' from the TA output directory to identify
    containers that have been released. Updates the 'released_flag' and
    'release_datetime' columns in the main container_data file for these
    containers and overwrites the file.

    Args:
        fc_id: Facility Center ID.
        ta_planning_timestamp: Timestamp used for the Tour Allocation run.
        base_ta_output_dir: Base directory containing TA outputs.
        container_data_path: Path to the container_data.csv file to be updated.
        logger: Optional logger instance. If None, a default logger is set up.

    Raises:
        FileNotFoundError: If tours_to_release.csv or container_data.csv is not found.
        ValueError: If required columns ('container_id') are missing in input files.
        Exception: For other potential errors during file processing.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("Starting update of container release status.")
    logger.info(f"FC ID: {fc_id}")
    logger.info(f"TA Timestamp: {ta_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Base TA Output Dir: {base_ta_output_dir}")
    logger.debug(f"Container Data File: {container_data_path}")

    try:
        # Construct TA output directory path
        ta_timestamp_str = ta_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        ta_output_dir = base_ta_output_dir / fc_id / ta_timestamp_str
        logger.debug(f"Target TA Output Directory: {ta_output_dir}")

        # Construct path to tours_to_release.csv
        tours_to_release_path = ta_output_dir / "tours_to_release.csv"
        logger.debug(f"Attempting to read released tours file: {tours_to_release_path}")

        if not tours_to_release_path.is_file():
            logger.error(f"Released tours file not found: {tours_to_release_path}")
            raise FileNotFoundError(f"Released tours file not found: {tours_to_release_path}")

        released_tours_df = pd.read_csv(tours_to_release_path)
        logger.debug(f"Read {len(released_tours_df)} rows from {tours_to_release_path}")

        # --- Get Released Container IDs ---
        if 'container_id' not in released_tours_df.columns:
            logger.error(f"'container_id' column not found in {tours_to_release_path}")
            raise ValueError(f"'container_id' column missing in {tours_to_release_path}")

        released_container_ids = released_tours_df['container_id'].unique().tolist()
        logger.info(f"Found {len(released_container_ids)} unique container IDs to release.")
        if not released_container_ids:
             logger.warning("No container IDs found in tours_to_release.csv. No updates will be made.")
             return # Nothing to update

        # --- Read and Update Container Data ---
        logger.debug(f"Attempting to read container data file: {container_data_path}")
        if not container_data_path.is_file():
            logger.error(f"Container data file not found: {container_data_path}")
            raise FileNotFoundError(f"Container data file not found: {container_data_path}")

        container_df = pd.read_csv(container_data_path)
        logger.debug(f"Read {len(container_df)} rows from {container_data_path}")

        if 'container_id' not in container_df.columns:
            logger.error(f"'container_id' column not found in {container_data_path}")
            raise ValueError(f"'container_id' column missing in {container_data_path}")

        # Ensure 'released_flag' column exists (as boolean)
        if 'released_flag' not in container_df.columns:
            logger.warning("'released_flag' column not found. Creating it with default False.")
            container_df['released_flag'] = False
        # Ensure it's boolean type for consistency
        container_df['released_flag'] = container_df['released_flag'].astype(bool)

        # Ensure 'release_datetime' column exists (as datetime)
        if 'release_datetime' not in container_df.columns:
            logger.warning("'release_datetime' column not found. Creating it with default NaT.")
            container_df['release_datetime'] = pd.NaT
        # Convert existing data to datetime if necessary, coercing errors
        container_df['release_datetime'] = pd.to_datetime(container_df['release_datetime'], errors='coerce')

        # Identify rows to update
        update_mask = container_df['container_id'].isin(released_container_ids)
        num_to_update = update_mask.sum()
        logger.info(f"Updating {num_to_update} rows in container data.")

        if num_to_update > 0:
            # Update columns for the matched rows
            container_df.loc[update_mask, 'released_flag'] = True
            container_df.loc[update_mask, 'release_datetime'] = ta_planning_timestamp

            # --- Save Updated Container Data (Overwrite) ---
            try:
                container_df.to_csv(container_data_path, index=False)
                logger.info(f"Successfully updated and saved container data to: {container_data_path}")
            except Exception as e:
                logger.error(f"Error writing updated container data to {container_data_path}: {e}", exc_info=True)
                raise # Re-raise the writing error
        else:
             logger.info("No matching container IDs found in the main container data file. No updates made.")

        logger.info("Update of container release status completed successfully.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Required file not found: {fnf_error}")
        raise
    except ValueError as val_error:
        logger.error(f"Data validation error: {val_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during container status update: {e}", exc_info=True)
        raise

# --- Function to Archive Released Tours and Update Tour Pool ---
def archive_and_update_tour_pool(
    fc_id: str,
    ta_planning_timestamp: datetime,
    base_ta_output_dir: Path,
    data_dir: Path,
    logger: logging.Logger = None
) -> None:
    """
    Archives tours_to_release.csv from a TA run and updates the consolidated
    tour pool files in the data directory by removing the released tours/containers.

    Args:
        fc_id: Facility Center ID.
        ta_planning_timestamp: Timestamp used for the Tour Allocation run.
        base_ta_output_dir: Base directory containing TA outputs.
        data_dir: Directory containing the consolidated ready-to-release tour files
                  (which will be updated) and where the archive will be stored.
        logger: Optional logger instance. If None, a default logger is set up.

    Raises:
        FileNotFoundError: If required input files (tours_to_release.csv or consolidated files) are missing.
        ValueError: If required columns are missing in input files.
        Exception: For other potential errors during file processing.
    """
    if logger is None:
        logger = setup_logging()

    logger.info(f"Starting archive and update of tour pool for TA run: {ta_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Data Directory: {data_dir}")

    try:
        # Construct TA output directory path
        ta_timestamp_str_dir = ta_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        ta_output_dir = base_ta_output_dir / fc_id / ta_timestamp_str_dir
        source_released_tours_path = ta_output_dir / "tours_to_release.csv"

        # --- Step 1: Archive tours_to_release.csv ---
        logger.debug(f"Checking for source file: {source_released_tours_path}")
        if not source_released_tours_path.is_file():
            logger.warning(f"Source file {source_released_tours_path} not found. No tours released in this run, or file missing. Skipping archive and update.")
            return # Nothing to archive or update

        archive_timestamp_str = ta_planning_timestamp.strftime('%Y%m%d_%H%M%S')
        archive_dest_path = data_dir / f"tours_to_release_{archive_timestamp_str}.csv"

        try:
            shutil.copy2(source_released_tours_path, archive_dest_path)
            logger.info(f"Successfully archived {source_released_tours_path} to {archive_dest_path}")
        except Exception as e:
            logger.error(f"Error archiving file {source_released_tours_path} to {archive_dest_path}: {e}", exc_info=True)
            raise # Re-raise archiving error

        # --- Step 2: Identify Released IDs ---
        logger.debug(f"Reading archived released tours file: {archive_dest_path}")
        try:
            released_df = pd.read_csv(archive_dest_path)
            if released_df.empty:
                logger.warning(f"Archived file {archive_dest_path} is empty. No tours/containers to remove from pool.")
                return # Nothing to update if the file is empty

            if 'tour_id' not in released_df.columns or 'container_id' not in released_df.columns:
                logger.error(f"Required columns ('tour_id', 'container_id') not found in {archive_dest_path}. Cannot update tour pool.")
                raise ValueError(f"Missing required columns in {archive_dest_path}")

            released_tour_ids = released_df['tour_id'].unique().tolist()
            released_container_ids = released_df['container_id'].unique().tolist()
            logger.info(f"Identified {len(released_tour_ids)} unique tour IDs and {len(released_container_ids)} unique container IDs to remove from pool.")

        except FileNotFoundError:
            # This shouldn't happen if copy succeeded, but handle defensively
            logger.error(f"Archived file {archive_dest_path} not found after supposedly copying. Cannot update tour pool.")
            raise
        except Exception as e:
            logger.error(f"Error reading archived file {archive_dest_path}: {e}", exc_info=True)
            raise

        # --- Step 3: Update Consolidated Files ---
        consolidated_files = {
            "aisle_ranges.csv": "tour_id",
            "container_assignments.csv": "container_id",
            "container_tours.csv": "tour_id",
            "pick_assignments.csv": "container_id"
        }

        for filename, id_column in consolidated_files.items():
            file_path = data_dir / filename
            logger.debug(f"Processing consolidated file: {file_path}")

            if not file_path.is_file():
                logger.warning(f"Consolidated file {file_path} not found. Skipping update for this file.")
                continue

            try:
                consolidated_df = pd.read_csv(file_path)
                if consolidated_df.empty:
                    logger.debug(f"Consolidated file {file_path} is empty. Nothing to update.")
                    continue

                if id_column not in consolidated_df.columns:
                     logger.warning(f"Required ID column '{id_column}' not found in {file_path}. Skipping update for this file.")
                     continue

                initial_rows = len(consolidated_df)

                # Determine which IDs to use for filtering
                ids_to_remove = released_tour_ids if id_column == 'tour_id' else released_container_ids

                if not ids_to_remove:
                    logger.debug(f"No relevant IDs ({id_column}) to remove for file {filename}. Skipping filtering.")
                    continue

                # Apply filtering
                filtered_df = consolidated_df[~consolidated_df[id_column].isin(ids_to_remove)]
                rows_removed = initial_rows - len(filtered_df)

                if rows_removed > 0:
                    # Overwrite the file with the filtered data
                    filtered_df.to_csv(file_path, index=False)
                    logger.debug(f"Updated {file_path}: Removed {rows_removed} rows based on released {id_column}s.")
                else:
                    logger.debug(f"No rows removed from {file_path}. All records remain.")

            except Exception as e:
                logger.error(f"Error processing or updating file {file_path}: {e}", exc_info=True)
                # Decide if we should continue with other files or stop
                # For now, log the error and continue with the next file

        logger.info(f"Archive and update of tour pool completed for TA run: {ta_planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    except FileNotFoundError as fnf_error:
        logger.error(f"Required file not found during archive/update: {fnf_error}")
        raise
    except ValueError as val_error:
        logger.error(f"Data validation error during archive/update: {val_error}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during archive/update: {e}", exc_info=True)
        raise

# --- Utility Function - Logger ---
class ErrorHandler(logging.Handler):
    """Custom handler that raises an exception when an error is logged."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            raise Exception(f"Error logged: {record.getMessage()}")

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Add the error handler
    error_handler = ErrorHandler()
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    return logger
