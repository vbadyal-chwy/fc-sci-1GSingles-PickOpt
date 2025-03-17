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

@dataclass
class ModelData:
    """Container for preprocessed model data"""
    tours: List[int]                          # Active tour IDs
    aisles: List[int]                         # Unique aisles
    tour_aisle_visits: Dict[Tuple[int, int], int]  # Tour-aisle visit matrix
    tour_lateness: Dict[int, float]           # Lateness by tour
    max_buffer_spots: int                     # Maximum number of buffer spots
    buffer_variability: float                 # Buffer variability factor

def prepare_model_data(
    tours_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    logger: logging.Logger = None
) -> ModelData:
    """
    Prepare data structures for optimization model.
    
    Parameters
    ----------
    tours_data : Dict[str, pd.DataFrame]
        Dictionary containing:
        - tour_metrics: Tour-level metrics DataFrame with columns [tour_id, num_containers, total_lateness, etc.]
        - pick_assignments: Pick assignments DataFrame with columns [tour_id, sku, aisle, quantity]
        - container_assignments: Container assignments DataFrame with columns [container_id, tour_id]
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
        logger = logging.getLogger(__name__)
        
    try:
        # Get active tours from tour metrics
        active_tours = []
        for _, tour in tours_data['tour_metrics'].iterrows():
            if tour['num_containers'] > 0:
                active_tours.append(tour['tour_id'])
        active_tours = sorted(active_tours)
        
        # Get aisles from pick assignments
        picks_df = tours_data['pick_assignments']
        aisles = picks_df[picks_df['tour_id'].isin(active_tours)]['aisle'].unique()
        aisles = sorted(list(aisles))
        
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
        
        # Calculate lateness by tour
        tour_lateness = {}
        for _, tour in tours_data['tour_metrics'].iterrows():
            if tour['tour_id'] in active_tours:
                tour_lateness[tour['tour_id']] = tour.get('total_lateness', 0.0)
        
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
            tour_lateness=tour_lateness,
            max_buffer_spots=max_buffer_spots,
            buffer_variability=buffer_variability
        )
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise