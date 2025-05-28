"""
Main entry point for tour allocation.

This module provides the main entry point for the tour allocation process,
including data loading, preprocessing, and orchestration of the solving process.
"""

from typing import Dict, Any, Optional
import logging
import pandas as pd

from .ta_solver import TourAllocationSolver, TourAllocationResult
from logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_allocation')

def run_tour_allocation(
    unassigned_tours: Dict,
    tours_to_release: int,
    pending_tours_by_aisle: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Optional[TourAllocationResult]:
    """
    Run the tour allocation process.
    
    Parameters
    ----------
    unassigned_tours : Dict
        Dictionary containing tour data for unassigned tours:
        {tour_id: {
            'containers': List[str],
            'picks': List[Dict],
            'metrics': Dict,
            'aisle_range': Dict
        }}
    tours_to_release : int
        Number of tours to release this iteration
    pending_tours_by_aisle : pd.DataFrame
        DataFrame containing pending tours by aisle
    config : Dict[str, Any]
        Configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    Optional[TourAllocationResult]
        Results from the allocation process
    """
    try:
        # Initialize solver
        solver = TourAllocationSolver(config, logger)
        
        # Convert unassigned tours data to required format
        tours_data = _prepare_tours_data(unassigned_tours)
        
        # Run solver
        result = solver.solve(tours_data, tours_to_release, pending_tours_by_aisle)
        
        # Log final summary
        if result:
            _log_final_summary(result, tours_to_release, logger)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in tour allocation process: {str(e)}")
        raise

def _prepare_tours_data(unassigned_tours: Dict) -> Dict[str, pd.DataFrame]:
    """
    Convert unassigned tours data into format required by solver.
    
    Parameters
    ----------
    unassigned_tours : Dict
        Dictionary containing tour data
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - tour_metrics: Tour-level metrics
        - pick_assignments: Individual pick assignments
        - container_assignments: Container-tour assignments
    """
    # Prepare tour metrics
    tour_metrics_data = []
    for tour_id, tour_data in unassigned_tours.items():
        metrics = {
            'tour_id': tour_id,
            'num_containers': len(tour_data['containers']),
            'num_picks': len(tour_data['picks']),
            'min_aisle': tour_data['aisle_range']['min_aisle'],
            'max_aisle': tour_data['aisle_range']['max_aisle'],
            'total_slack': tour_data['aisle_range']['total_slack']
        }
        if 'metrics' in tour_data:
            metrics.update(tour_data['metrics'])
        tour_metrics_data.append(metrics)
    
    # Prepare pick assignments
    pick_assignments_data = []
    for tour_id, tour_data in unassigned_tours.items():
        for pick in tour_data['picks']:
            pick['tour_id'] = tour_id
            pick_assignments_data.append(pick)
            
    # Prepare container assignments
    container_assignments_data = []
    for tour_id, tour_data in unassigned_tours.items():
        for container_id in tour_data['containers']:
            container_assignments_data.append({
                'container_id': container_id,
                'tour_id': tour_id
            })
            
    return {
        'tour_metrics': pd.DataFrame(tour_metrics_data),
        'pick_assignments': pd.DataFrame(pick_assignments_data),
        'container_assignments': pd.DataFrame(container_assignments_data)
    }

def _log_final_summary(
    result: TourAllocationResult,
    max_buffer_spots: int,
    logger: logging.Logger
) -> None:
    """
    Log final summary of tour allocation results.
    
    Parameters
    ----------
    result : TourAllocationResult
        Results from the allocation process
    max_buffer_spots : int
        Maximum number of buffer spots available
    logger : logging.Logger
        Logger instance
    """
    # Calculate metrics
    total_tours = len(result.buffer_assignments)
    empty_spots = max_buffer_spots - total_tours
    buffer_utilization = (total_tours / max_buffer_spots) * 100 if max_buffer_spots > 0 else 0
    max_concurrency = max(len(buffers) for buffers in result.aisle_assignments.values())
    
    # Log summary
    logger.info("\nFinal Tour Allocation Summary:")
    logger.info("=" * 50)
    logger.info(f"Total tours allocated: {total_tours}")
    logger.info(f"Empty buffer spots: {empty_spots} out of {max_buffer_spots}")
    logger.info(f"Buffer utilization: {buffer_utilization:.1f}%")
    logger.info(f"Maximum aisle concurrency: {max_concurrency}")