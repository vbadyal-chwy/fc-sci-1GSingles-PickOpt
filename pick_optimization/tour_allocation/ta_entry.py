"""
Entry point for tour allocation in containerized mode.
"""

from typing import Union, Optional
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from .utils import normalize_path, load_model_config
from .ta_data_exchange import load_ta_input_data, write_ta_output_data
from .ta_main import run_tour_allocation
from .tour_buffer import TourBuffer

# --- Entrypoint ---

# Get module-specific logger
logger = logging.getLogger(__name__)

def run_tour_allocation_entrypoint(
    fc_id: str,
    planning_timestamp: Union[str, datetime],  
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_tours: Optional[int]
) -> None:
    """
    Run tour allocation, loading from and saving to specified directories.

    Parameters
    ----------
    fc_id : str
        Fulfillment center ID.
    planning_timestamp : Union[str, datetime]
        Planning timestamp (ISO format string or datetime object).
    input_dir : Union[str, Path]
        Directory containing input CSV files.
    output_dir : Union[str, Path]
        Directory to write output CSV files.
    target_tours : Optional[int]
        Optional override for the number of tours to release.
    """
    try:
        # Validate and normalize inputs
        input_dir = normalize_path(input_dir)
        output_dir = normalize_path(output_dir)

    
        if isinstance(planning_timestamp, str):
            try:
                planning_timestamp = datetime.fromisoformat(planning_timestamp)
            except ValueError as e:
                raise ValueError(
                    f"Invalid planning_timestamp format: {planning_timestamp}. "
                    f"Expected ISO format (e.g., 2024-05-15T14:30:00)"
                ) from e

        logger.info("Starting Tour Allocation entrypoint.")
        logger.info(f"FC ID: {fc_id}")
        logger.info(f"Planning Timestamp: {planning_timestamp}")
        logger.debug(f"Using Input Dir: {input_dir}")
        logger.debug(f"Using Output Dir: {output_dir}")

        # Load config from input directory using utility function
        try:
            config = load_model_config(input_dir)
            logger.info(f"Loaded config using load_model_config from input_dir: {input_dir}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found in {input_dir}. Exiting.")
            raise
        except Exception as config_load_error:
            logger.error(f"Error loading configuration from {input_dir}: {config_load_error}")
            raise

        # Inject target tours override if provided
        if target_tours is not None:
            config['target_tours'] = target_tours
            logger.info(f"Tours to Release (count): {target_tours}")
        else:
            logger.info("No target tours override provided.")

        container_assignments_df, pick_assignments_df, aisle_ranges_df, pending_tours_by_aisle_df, container_tours_df = \
            load_ta_input_data(input_dir, logger)

        if container_tours_df.empty:
            logger.warning("No container tours found. Skipping tour allocation.")
            write_ta_output_data(output_dir, None, {'status': 'No container tours found'}, container_tours_df, logger)
            return

        tour_buffer = TourBuffer(config, logger)

        # Prepare unassigned tours data
        logger.debug("Preparing unassigned tours data from loaded dataframes.")
        unassigned_tours = {}
        for tour_id in container_assignments_df['tour_id'].unique():
            tour_containers = container_assignments_df[
                container_assignments_df['tour_id'] == tour_id
            ]['container_id'].tolist()
            tour_picks = pick_assignments_df[
                pick_assignments_df['container_id'].isin(tour_containers)
            ].to_dict('records')
            try:
                tour_aisle_range = aisle_ranges_df[aisle_ranges_df['tour_id'] == tour_id].iloc[0].to_dict()
            except IndexError:
                 logger.error(f"Aisle range data missing for tour_id: {tour_id}. Skipping tour.")
                 continue 

            unassigned_tours[tour_id] = {
                'containers': tour_containers,
                'picks': tour_picks,
                'aisle_range': tour_aisle_range
            }
        logger.info(f"Prepared {len(unassigned_tours)} unassigned tours.")

        # Determine number of tours to release
        tours_to_release = target_tours
        current_buffer = 0 

        # Initialize allocation metrics dictionary
        allocation_metrics = {
            'total_tours_allocated': 0,
            'current_buffer_size': current_buffer, 
            'tours_released_request': tours_to_release,
            'max_aisle_concurrency': 0,
            'status': 'Started'
        }

        if tours_to_release <= 0:
            logger.warning(f"No tours to release (target: {tours_to_release}). Stopping allocation.")
            allocation_metrics['status'] = 'No tours to release'
            write_ta_output_data(output_dir, None, allocation_metrics, container_tours_df, logger)
            return 

        # Run tour allocation main logic
        logger.info(f"Running core tour allocation for {tours_to_release} tours...")
        result = run_tour_allocation(
            unassigned_tours=unassigned_tours,
            tours_to_release=tours_to_release,
            pending_tours_by_aisle=pending_tours_by_aisle_df,
            config=config,
            logger=logger
        )

        if result:
            allocation_metrics['total_tours_allocated'] = len(result.buffer_assignments)
            if result.aisle_assignments:
                 allocation_metrics['max_aisle_concurrency'] = max(len(tours) for tours in result.aisle_assignments.values())
            allocation_metrics['status'] = 'Completed'
            logger.info("Tour allocation core logic finished successfully.")
        else:
            logger.warning("No solution found by tour allocation core logic.")
            allocation_metrics['status'] = 'No solution found'

        # Write output data using helper
        write_ta_output_data(output_dir, result, allocation_metrics, container_tours_df, logger)

        logger.info("Tour Allocation entrypoint finished.")

    except FileNotFoundError as fnf_error:
        logger.error(f"Input file not found: {str(fnf_error)}")
        try:
            write_ta_output_data(output_dir, None, {'status': f'Input file error: {fnf_error}'}, pd.DataFrame(), logger)
        except Exception as write_err:
            logger.error(f"Failed to write error status metrics: {write_err}")
        raise
    except Exception as e:
        logger.error(f"Error in tour allocation entrypoint: {str(e)}", exc_info=True)
        try:
            write_ta_output_data(output_dir, None, {'status': f'Runtime error: {e}'}, pd.DataFrame(), logger) 
        except Exception as write_err:
            logger.error(f"Failed to write error status metrics: {write_err}")
        raise 