"""
Data preparation and preprocessing for tour allocation optimization.

This module handles all data preparation tasks including:
- Tour data preprocessing
- Aisle mapping
- Buffer spot configuration
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import logging
from pick_optimization.utils.logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_allocation')

@dataclass
class ModelData:
    """Container for preprocessed model data"""
    tours: List[int]                          # Active tour IDs
    aisles: List[int]                         # Unique aisles
    tour_aisle_visits: Dict[Tuple[int, int], int]  # Tour-aisle visit matrix
    total_slack: Dict[int, float]           # Total slack by tour
    max_buffer_spots: int                     # Maximum number of buffer spots
    buffer_variability: float                 # Buffer variability factor
    aisle_concurrency: Dict[int, int]         # Number of concurrent tours per aisle

def prepare_model_data(
    tours_data: Dict[str, pd.DataFrame],
    pending_tours_by_aisle: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger = None
) -> ModelData:
    """
    Prepare data structures for optimization model.
    
    Parameters
    ----------
    tours_data : Dict[str, pd.DataFrame]
        Dictionary containing:
        - tour_metrics: Tour-level metrics DataFrame with columns [tour_id, num_containers, total_slack, etc.]
        - pick_assignments: Pick assignments DataFrame with columns [tour_id, sku, aisle, quantity]
        - container_assignments: Container assignments DataFrame with columns [container_id, tour_id]
    pending_tours_by_aisle: pd.DataFrame
        DataFrame containing pending tours by aisle with columns [wh_id, snapshot_datetime, aisle, tour_count, quantity]
    config : Dict[str, Any]
        Configuration dictionary
    logger : Optional[logging.Logger]
        Logger instance for logging messages
        
    Returns
    -------
    ModelData
        Preprocessed data ready for model building
    """
    if logger is None:
        logger = get_logger(__name__, 'tour_allocation')
        
    try:
        # Get active tours from tour metrics
        active_tours = []
        for _, tour in tours_data['tour_metrics'].iterrows():
            if tour['num_containers'] > 0:
                active_tours.append(tour['tour_id'])
        active_tours = sorted(active_tours)
        
        # Get aisles from pick assignments
        picks_df = tours_data['pick_assignments']
        pool_aisles = picks_df[picks_df['tour_id'].isin(active_tours)]['aisle'].unique()
        
        # Get aisles from pending tours
        pending_aisles = pending_tours_by_aisle['aisle'].unique()
        
        # Convert both aisle arrays to integers to ensure data type consistency
        # Handle the case where pending_aisles are floats (like 11.0) and pool_aisles are ints (like 11)
        
        # Convert pool_aisles to integers (they're numpy arrays from .unique(), so convert to Series first)
        pool_aisles = pd.Series(pool_aisles)
        pool_aisles = pd.to_numeric(pool_aisles, errors='coerce').dropna().astype(int)
        
        # Convert pending_aisles to integers, handling the float case properly
        pending_aisles = pd.Series(pending_aisles)
        pending_aisles = pd.to_numeric(pending_aisles, errors='coerce').dropna().astype(int)
        
        logger.debug(f"Pool aisles (int): {sorted(pool_aisles.tolist())}")
        logger.debug(f"Pending aisles (int): {sorted(pending_aisles.tolist())}")
        
        # Combine and get unique sorted list of all aisles - now both are guaranteed to be integers
        aisles = sorted(list(set(pool_aisles.tolist()) | set(pending_aisles.tolist())))
        
        # Create tour-aisle visit matrix
        tour_aisle_visits = {}
        for tour_id in active_tours:
            # Initialize all visits to 0
            for aisle in aisles:
                tour_aisle_visits[tour_id, aisle] = 0
                
        # Set visits based on picks
        tour_picks = picks_df[picks_df['tour_id'].isin(active_tours)]
        for _, pick in tour_picks.iterrows():
            tour_aisle_visits[pick['tour_id'], pick['aisle']] = 1
        
        # Calculate total slack by tour
        total_slack = {}
        for _, tour in tours_data['tour_metrics'].iterrows():
            if tour['tour_id'] in active_tours:
                total_slack[tour['tour_id']] = tour.get('total_slack')
        
        # Create aisle concurrency dictionary from pending_tours_by_aisle
        aisle_concurrency = {}
        for _, row in pending_tours_by_aisle.iterrows():
            aisle = int(row['aisle'])
            tour_count = int(row['tour_count'])
            aisle_concurrency[aisle] = tour_count
        
        # Extract buffer configuration
        max_buffer_spots = config['tour_allocation']['max_buffer_spots']
        buffer_variability = config['tour_allocation']['buffer_variability_factor']
        
        # Log preprocessing statistics
        logger.info(f"Processed {len(active_tours)} active tours")
        logger.info(f"Maximum buffer spots: {max_buffer_spots}")
        
        return ModelData(
            tours=active_tours,
            aisles=aisles,
            tour_aisle_visits=tour_aisle_visits,
            total_slack=total_slack,
            max_buffer_spots=max_buffer_spots,
            buffer_variability=buffer_variability,
            aisle_concurrency=aisle_concurrency
        )
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise