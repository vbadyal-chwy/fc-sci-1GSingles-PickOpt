"""
Main solver interface for tour allocation.

This module provides the high-level interface for solving the tour allocation problem,
including data preparation, model building, and solution extraction.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate

from .ta_data import ModelData, prepare_model_data
from .ta_model import TourAllocationModel
from logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_allocation')

@dataclass
class TourAllocationResult:
    """Container for tour allocation results."""
    buffer_assignments: Dict[int, int]  # TourID -> BufferSpotID
    aisle_assignments: Dict[int, List[int]]  # Aisle -> List[BufferSpotID]
    metrics: Dict[str, float]

class TourAllocationSolver:
    """Main solver interface for tour allocation problem."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the solver.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        logger : logging.Logger
            Logger instance
        """
        self.config = config
        self.logger = logger
        
    def solve(self, 
              tours_data: Dict[str, pd.DataFrame],
              tours_to_release: int,
              pending_tours_by_aisle: pd.DataFrame) -> Optional[TourAllocationResult]:
        """
        Solve the tour allocation problem.
        
        Parameters
        ----------
        tours_data : Dict[str, pd.DataFrame]
            Dictionary containing:
            - tour_metrics: Tour-level metrics
            - pick_assignments: Individual pick assignments
            - container_assignments: Container-tour assignments
        tours_to_release : int
            Number of tours to release this iteration
        pending_tours_by_aisle : pd.DataFrame
            DataFrame containing pending tours by aisle
        Returns
        -------
        Optional[TourAllocationResult]
            Results from solving the allocation problem
        """
        try:
            # Update max buffer spots in config
            self.config['tour_allocation']['max_buffer_spots'] = tours_to_release
            
            # Prepare data
            model_data = prepare_model_data(tours_data, pending_tours_by_aisle, self.config, self.logger)
            
            # Build and solve model
            model = TourAllocationModel(model_data, self.config, self.logger)
            model.build()
            solution = model.solve()
            
            if solution:
                # Create result object
                result = TourAllocationResult(
                    buffer_assignments=solution['tour_assignments'],
                    aisle_assignments=solution['aisle_assignments'],
                    metrics=solution['metrics']
                )
                
                # Generate summary
                self._generate_summary(result, model_data)
                
                return result
            else:
                self.logger.warning("No solution found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in tour allocation solving: {str(e)}")
            raise
            
    def _generate_summary(self, result: TourAllocationResult, model_data: ModelData) -> None:
        """
        Generate and log summary of allocation results.
        
        Parameters
        ----------
        result : TourAllocationResult
            Results from solving the allocation problem
        """
        # 1. Tour Allocation Summary
        tour_summary = []
        for tour_id, buffer_id in sorted(result.buffer_assignments.items()):
            tour_aisles = [
                a for a, buffers in result.aisle_assignments.items()
                if buffer_id in buffers
            ]
            tour_summary.append([
                f"Tour {tour_id}",
                f"Buffer {buffer_id}",
                len(tour_aisles),
                f"{min(tour_aisles)}-{max(tour_aisles)}" if tour_aisles else "None"
            ])
        
        self.logger.info("\nTour Allocation Summary:")
        self.logger.info("\n" + tabulate(
            tour_summary,
            headers=['Tour', 'Assigned Buffer', 'Aisle Count', 'Aisle Range'],
            tablefmt='grid'
        ))
        
        # 2. Aisle Concurrency Summary
        aisle_summary = []
        for aisle, buffers in sorted(result.aisle_assignments.items()):
            if buffers:  # Only show aisles with assigned buffers
                # Get pending tour count from aisle_concurrency
                pending_tours = model_data.aisle_concurrency.get(aisle, 0)
                # Calculate new tour count
                new_tours = len(buffers)
                # Calculate total tour count
                total_tours = pending_tours + new_tours
                
                aisle_summary.append([
                    f"Aisle {aisle}",
                    total_tours,
                    pending_tours,
                    new_tours,
                    ','.join(map(str, sorted(buffers)))
                ])
        
        if aisle_summary:
            self.logger.info("\nAisle Concurrency Summary:")
            self.logger.info("\n" + tabulate(
                aisle_summary,
                headers=['Aisle', 'Total Tours', 'Pending Tours', 'New Tours', 'New Assigned Tour IDs'],
                tablefmt='grid'
            ))
            
        # 3. Buffer Utilization Summary
        buffer_summary = []
        buffer_to_tours = {}
        for tour_id, buffer_id in result.buffer_assignments.items():
            if buffer_id not in buffer_to_tours:
                buffer_to_tours[buffer_id] = []
            buffer_to_tours[buffer_id].append(tour_id)
            
        for buffer_id in range(self.config['tour_allocation']['max_buffer_spots']):
            assigned_tours = buffer_to_tours.get(buffer_id, [])
            if assigned_tours:
                buffer_summary.append([
                    f"Buffer {buffer_id}",
                    len(assigned_tours),
                    ','.join(map(str, sorted(assigned_tours)))
                ])
            else:
                buffer_summary.append([
                    f"Buffer {buffer_id}",
                    0,
                    "Empty"
                ])
                
        self.logger.info("\nBuffer Utilization Summary:")
        self.logger.info("\n" + tabulate(
            buffer_summary,
            headers=['Buffer', 'Tour Count', 'Assigned Tour IDs'],
            tablefmt='grid'
        ))