"""
Module for handling data loading and saving for Tour Allocation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

def load_ta_input_data(
    input_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads required input CSV files for Tour Allocation."""
    container_assignments_path = input_dir / 'container_assignments.csv'
    pick_assignments_path = input_dir / 'pick_assignments.csv'
    aisle_ranges_path = input_dir / 'aisle_ranges.csv'
    pending_tours_by_aisle_path = input_dir / 'pending_tours_by_aisle.csv'
    container_tours_path = input_dir / 'container_tours.csv'

    if not container_assignments_path.is_file():
        raise FileNotFoundError(f"Container assignments file not found: {container_assignments_path}")
    if not pick_assignments_path.is_file():
        raise FileNotFoundError(f"Pick assignments file not found: {pick_assignments_path}")
    if not aisle_ranges_path.is_file():
        raise FileNotFoundError(f"Aisle ranges file not found: {aisle_ranges_path}")
    if not pending_tours_by_aisle_path.is_file():
        raise FileNotFoundError(f"Pending tours by aisle file not found: {pending_tours_by_aisle_path}")
    if not container_tours_path.is_file():
        raise FileNotFoundError(f"Container tours file not found: {container_tours_path}")

    logger.debug(f"Reading container assignments from: {container_assignments_path}")
    container_assignments_df = pd.read_csv(container_assignments_path)
    logger.debug(f"Reading pick assignments from: {pick_assignments_path}")
    pick_assignments_df = pd.read_csv(pick_assignments_path)
    logger.debug(f"Reading aisle ranges from: {aisle_ranges_path}")
    aisle_ranges_df = pd.read_csv(aisle_ranges_path)
    logger.debug(f"Reading pending tours by aisle from: {pending_tours_by_aisle_path}")
    pending_tours_by_aisle_df = pd.read_csv(pending_tours_by_aisle_path)
    logger.debug(f"Reading container tours from: {container_tours_path}")
    container_tours_df = pd.read_csv(container_tours_path)

    
    logger.info("Tour Allocation input data loaded.")
    return container_assignments_df, pick_assignments_df, aisle_ranges_df, pending_tours_by_aisle_df, container_tours_df

def write_ta_output_data(
    output_dir: Path,
    result: Optional[Dict],
    allocation_metrics: Dict,
    container_tours_df: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """Writes the Tour Allocation output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_assignments_written = False
    allocated_tour_ids = []

    if result:
        buffer_assignments_path = output_dir / 'buffer_assignments.csv'
        buffer_assignments = []
        if result is not None and hasattr(result, 'buffer_assignments') and result.buffer_assignments:
             allocated_tour_ids = list(result.buffer_assignments.keys())
             for tour_id, buffer_id in result.buffer_assignments.items():
                 buffer_assignments.append({
                     'tour_id': tour_id,
                     'buffer_id': buffer_id
                 })
             pd.DataFrame(buffer_assignments).to_csv(buffer_assignments_path, index=False)
             logger.debug(f"Buffer assignments written to: {buffer_assignments_path}")
             buffer_assignments_written = True
        else:
             logger.warning("No buffer assignments found in result to write.")


        aisle_assignments_path = output_dir / 'aisle_assignments.csv'
        aisle_assignments = []
        if result is not None and hasattr(result, 'aisle_assignments') and result.aisle_assignments:
             for aisle, tour_ids in result.aisle_assignments.items():
                 for tour_id in tour_ids:
                     aisle_assignments.append({
                         'aisle': aisle,
                         'tour_id': tour_id
                     })
             pd.DataFrame(aisle_assignments).to_csv(aisle_assignments_path, index=False)
             logger.debug(f"Aisle assignments written to: {aisle_assignments_path}")
        else:
             logger.warning("No aisle assignments found in result to write.")

        if buffer_assignments_written:
            tours_to_release_path = output_dir / 'tours_to_release.csv'
            tours_to_release_df = container_tours_df[
                container_tours_df['tour_id'].isin(allocated_tour_ids)
            ]
            tours_to_release_df.to_csv(tours_to_release_path, index=False)
            logger.debug(f"Tours to release details written to: {tours_to_release_path}")
        else:
            logger.warning("Buffer assignments were not written, skipping tours_to_release.csv file writing.")

    else:
        logger.warning("No allocation result object provided, skipping buffer, aisle assignment, and tours_to_release file writing.")

    allocation_metrics_path = output_dir / 'allocation_metrics.csv'
    pd.DataFrame([allocation_metrics]).to_csv(allocation_metrics_path, index=False)
    logger.debug(f"Allocation metrics written to: {allocation_metrics_path}")
    logger.info("Tour Allocation output files written.")

