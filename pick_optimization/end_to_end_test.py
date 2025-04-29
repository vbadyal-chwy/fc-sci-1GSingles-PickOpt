"""
End-to-end test script for running Tour Formation (TF) and Tour Allocation (TA)
sequentially over a time period.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

# --- Platform Utils ---
try:
    from platform_utils import (
        create_tour_formation_inputs,
        create_tour_allocation_inputs,
        update_container_release_status,
        create_ready_to_release_tours,
        archive_and_update_tour_pool,
        get_project_root,
        setup_logging
    )
except ImportError:
    print("Error: Make sure platform_utils.py is accessible.")
    def get_project_root(): return Path(__file__).parent
    def setup_logging():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    def create_tour_formation_inputs(*args, **kwargs): pass
    def create_tour_allocation_inputs(*args, **kwargs): pass
    def update_container_release_status(*args, **kwargs): pass
    def create_ready_to_release_tours(*args, **kwargs): pass
    def archive_and_update_tour_pool(*args, **kwargs): pass


# --- Model Entrypoints ---
try:
    from tour_formation.tf_entry import run_tour_formation_entrypoint
except ImportError:
    print("Error: Make sure tour_formation package and tf_entry.py are accessible.")
    def run_tour_formation_entrypoint(*args, **kwargs): return None

try:
    from tour_allocation.ta_entry import run_tour_allocation_entrypoint
except ImportError:
    print("Error: Make sure tour_allocation package and ta_entry.py are accessible.")
    def run_tour_allocation_entrypoint(*args, **kwargs): pass 


# --- Configuration ---
FC_ID = 'AVP1'
START_DATETIME_STR = '2025-03-06 14:00:00'
END_DATETIME_STR = '2025-03-06 14:40:00'
TF_INTERVAL_MINUTES = 30
TA_INTERVAL_MINUTES = 5
LABOR_HEADCOUNT = 50
TARGET_TOURS = 5
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
TIMESTAMP_DIR_FORMAT = '%Y%m%d_%H%M%S'

# --- Paths ---
PROJECT_ROOT = get_project_root()
BASE_INPUT_DIR = PROJECT_ROOT / "pick_optimization" / "input"
BASE_OUTPUT_DIR = PROJECT_ROOT / "pick_optimization" / "output"
BASE_WORKING_DIR = PROJECT_ROOT / "pick_optimization" / "working"
SOURCE_DATA_DIR = PROJECT_ROOT / "pick_optimization" /"data"
SOURCE_CONTAINER_DATA_PATH = SOURCE_DATA_DIR / "container_data.csv"
SOURCE_SLOTBOOK_DATA_PATH = SOURCE_DATA_DIR / "slotbook_data.csv"
SOURCE_TOUR_FORMATION_CONFIG_PATH = SOURCE_DATA_DIR / "tour_formation_config.yaml"
SOURCE_TOUR_ALLOCATION_CONFIG_PATH = SOURCE_DATA_DIR / "tour_allocation_config.yaml"

# --- Utility Function - Ensure Directory Exists ---
def ensure_dir_exists(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    Returns the Path object.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    """Runs the end-to-end TF and TA process simulation."""
    logger = setup_logging()
    logger.info("--- Starting End-to-End Pick Planning Simulation ---")
    logger.info(f"FC ID: {FC_ID}")
    logger.info(f"Time Range: {START_DATETIME_STR} to {END_DATETIME_STR}")
    logger.info(f"TF Interval: {TF_INTERVAL_MINUTES} min, TA Interval: {TA_INTERVAL_MINUTES} min")
    logger.info(f"Source Container Data: {SOURCE_CONTAINER_DATA_PATH}")
    logger.info(f"Source Slotbook Data: {SOURCE_SLOTBOOK_DATA_PATH}")

    if not SOURCE_CONTAINER_DATA_PATH.is_file() or not SOURCE_SLOTBOOK_DATA_PATH.is_file():
         logger.error(f"Source data files not found. Searched in: {SOURCE_DATA_DIR}")
         logger.error("Please ensure container_data.csv and slotbook_data.csv exist in the expected location.")
         return 

    try:
        # --- Time Initialization ---
        start_time = datetime.strptime(START_DATETIME_STR, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(END_DATETIME_STR, TIMESTAMP_FORMAT)
        tf_interval = timedelta(minutes=TF_INTERVAL_MINUTES)
        ta_interval = timedelta(minutes=TA_INTERVAL_MINUTES)

        current_time = start_time
        next_tf_time = start_time
        next_ta_time = start_time
        last_successful_tf_time = None

        # --- Main Simulation Loop ---
        while current_time <= end_time:
            logger.info(f"--- Simulation Time: {current_time.strftime(TIMESTAMP_FORMAT)} ---")

            run_tf_this_iteration = (current_time >= next_tf_time)
            run_ta_this_iteration = (current_time >= next_ta_time)

            # --- Execute Tour Formation (TF) ---
            if run_tf_this_iteration:
                logger.info(f"*** Running Tour Formation at {current_time.strftime(TIMESTAMP_FORMAT)} ***")
                tf_timestamp_str = current_time.strftime(TIMESTAMP_DIR_FORMAT)
                tf_input_dir = BASE_INPUT_DIR / FC_ID / tf_timestamp_str
                tf_output_dir = BASE_OUTPUT_DIR / FC_ID / tf_timestamp_str
                tf_working_dir = BASE_WORKING_DIR / FC_ID

                try:
                    # 1. Create TF Inputs
                    logger.info("\n\n=== Step 1: Creating TF Inputs ===")
                    tf_input_dir = create_tour_formation_inputs(
                        fc_id=FC_ID,
                        tf_planning_timestamp=current_time,
                        base_input_dir=BASE_INPUT_DIR,
                        container_data_path=SOURCE_CONTAINER_DATA_PATH,
                        slotbook_data_path=SOURCE_SLOTBOOK_DATA_PATH,
                        tour_formation_config_path=SOURCE_TOUR_FORMATION_CONFIG_PATH,
                        logger=logger
                    )
                    logger.info(f"TF Inputs created in: {tf_input_dir}")

                    # 2. Generate Clusters
                    logger.info("\n\n=== Step 2: Running TF - Generate Clusters ===")
                    ensure_dir_exists(tf_working_dir)
                    run_tour_formation_entrypoint(
                        mode='generate_clusters',
                        fc_id=FC_ID,
                        planning_timestamp=current_time,
                        input_dir=str(tf_input_dir),
                        output_dir=str(tf_output_dir), 
                        working_dir=str(tf_working_dir),
                        labor_headcount=LABOR_HEADCOUNT
                        # cluster_id is not needed for generate_clusters
                    )
                    logger.info("TF Generate Clusters completed.")

                    # 3. Solve Clusters
                    logger.info("\n\n=== Step 3: Running TF - Solve Clusters ===")
                    ensure_dir_exists(tf_output_dir)

                    # Read metadata to find cluster IDs
                    metadata_path = tf_output_dir / "clustering_metadata.csv"
                    if metadata_path.is_file():
                        metadata_df = pd.read_csv(metadata_path)
                        if 'cluster_id' in metadata_df.columns:
                            cluster_ids = metadata_df['cluster_id'].unique().tolist()
                            logger.info(f"Found {len(cluster_ids)} clusters to solve: {cluster_ids}")

                            for cluster_id in cluster_ids:
                                logger.info(f"Solving Cluster ID: {cluster_id}")
                                run_tour_formation_entrypoint(
                                    mode='solve_cluster',
                                    fc_id=FC_ID,
                                    planning_timestamp=current_time,
                                    input_dir=str(tf_input_dir),
                                    output_dir=str(tf_output_dir),
                                    working_dir=str(tf_working_dir),
                                    labor_headcount=LABOR_HEADCOUNT,
                                    cluster_id=int(cluster_id)
                                )
                                logger.info(f"Cluster {cluster_id} solved.")
                        else:
                            logger.error(f"'cluster_id' column not found in {metadata_path}. Cannot solve clusters.")
                    else:
                         logger.error(f"Clustering metadata file not found: {metadata_path}. Cannot solve clusters.")


                    logger.info(f"*** Tour Formation completed successfully for {current_time.strftime(TIMESTAMP_FORMAT)} ***")
                    last_successful_tf_time = current_time
                    next_tf_time += tf_interval

                    # Create ready-to-release tours after successful TF
                    logger.info("\n\n=== Step 3.5: Creating ready-to-release tours from TF outputs ===")
                    create_ready_to_release_tours(
                        fc_id=FC_ID,
                        tf_planning_timestamp=current_time,
                        base_tf_output_dir=BASE_OUTPUT_DIR,
                        data_dir=SOURCE_DATA_DIR,  
                        logger=logger
                    )
                    logger.info("Ready-to-release tours created successfully")

                except Exception as e:
                    logger.error(f"!!! Tour Formation failed at {current_time.strftime(TIMESTAMP_FORMAT)}: {e}", exc_info=True)
                    next_tf_time += tf_interval


            # --- Execute Tour Allocation (TA) ---
            if run_ta_this_iteration:
                logger.info(f"--- Running Tour Allocation at {current_time.strftime(TIMESTAMP_FORMAT)} ---")
                if last_successful_tf_time is None:
                    logger.warning(f"Skipping TA at {current_time.strftime(TIMESTAMP_FORMAT)} because no successful TF run has occurred yet.")
                else:
                    logger.info(f"Using TF outputs from: {last_successful_tf_time.strftime(TIMESTAMP_FORMAT)}")
                    ta_timestamp_str = current_time.strftime(TIMESTAMP_DIR_FORMAT)
                    last_tf_timestamp_str = last_successful_tf_time.strftime(TIMESTAMP_DIR_FORMAT)

                    ta_input_dir = BASE_INPUT_DIR / FC_ID / ta_timestamp_str
                    ta_output_dir = BASE_OUTPUT_DIR / FC_ID / ta_timestamp_str
                    tf_output_dir_for_ta = BASE_OUTPUT_DIR / FC_ID / last_tf_timestamp_str

                    try:
                        # 4. Create TA Inputs
                        logger.info("\n\n=== Step 4: Creating TA Inputs ===")
                        create_tour_allocation_inputs(
                             fc_id=FC_ID,
                             ta_planning_timestamp=current_time,
                             base_ta_input_dir=BASE_INPUT_DIR,
                             tour_allocation_config_path=SOURCE_DATA_DIR / "tour_allocation_config.yaml", 
                             data_dir=SOURCE_DATA_DIR, 
                             logger=logger
                         )
                        logger.info(f"TA Inputs created in: {ta_input_dir}")

                        # 5. Run TA Entrypoint
                        logger.info("\n\n=== Step 5: Running TA Entrypoint ===")
                        ensure_dir_exists(ta_output_dir)
                        run_tour_allocation_entrypoint(
                            fc_id=FC_ID,
                            planning_timestamp=current_time,
                            input_dir=str(ta_input_dir),
                            output_dir=str(ta_output_dir),
                            target_tours=TARGET_TOURS
                        )
                        logger.info("TA Entrypoint completed.")

                        # 6. Update Container Release Status
                        logger.info("\n\n=== Step 6: Updating Container Release Status ===")
                        update_container_release_status(
                            fc_id=FC_ID,
                            ta_planning_timestamp=current_time,
                            base_ta_output_dir=BASE_OUTPUT_DIR,
                            container_data_path=SOURCE_CONTAINER_DATA_PATH, 
                            logger=logger
                        )
                        logger.info("Container Release Status Updated.")

                        # 7. Archive Released Tours and Update Tour Pool
                        logger.info("\n\n=== Step 7: Archiving released tours and updating tour pool ===")
                        archive_and_update_tour_pool(
                            fc_id=FC_ID,
                            ta_planning_timestamp=current_time,
                            base_ta_output_dir=BASE_OUTPUT_DIR, 
                            data_dir=SOURCE_DATA_DIR,     
                            logger=logger
                        )
                        logger.info("Tour pool archived and updated.")

                        logger.info(f"--- Tour Allocation completed successfully for {current_time.strftime(TIMESTAMP_FORMAT)} ---")

                    except Exception as e:
                         logger.error(f"!!! Tour Allocation failed at {current_time.strftime(TIMESTAMP_FORMAT)}: {e}", exc_info=True)

                next_ta_time += ta_interval

            # --- Advance Simulation Time ---
            # Move to the next event time
            next_event_time = min(next_tf_time, next_ta_time)

            if next_event_time > end_time and current_time >= end_time:
                 # If the next event is past the end time, and we've processed the end time, break
                 break
            elif next_event_time > current_time:
                 # Only advance if the next event is actually in the future
                 current_time = next_event_time
            else:
                 current_time += min(tf_interval, ta_interval)
                 if not run_tf_this_iteration and not run_ta_this_iteration:
                     logger.warning("Advancing time by smallest interval as no models ran.")
                     current_time += min(tf_interval, ta_interval) # Ensure loop progresses


        logger.info("--- End-to-End Pick Planning Simulation Finished ---")

    except ValueError as e:
        logger.error(f"Configuration error (e.g., timestamp format): {e}")
    except FileNotFoundError as e:
         logger.error(f"File not found during setup or execution: {e}")
    except ImportError as e:
         logger.error(f"Failed to import required modules: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the simulation: {e}", exc_info=True)


if __name__ == "__main__":
    # Before running, ensure:
    # 1. `platform_utils.py`, `tour_formation` package, `tour_allocation` package are accessible.
    # 2. Source data files exist at `SOURCE_DATA_DIR / <filename>.csv`.
    # 3. Required directories (input, output, working) can be created.
    main() 