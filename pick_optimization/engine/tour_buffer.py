"""
Module for managing tour buffer and release calculations.

This module provides functionality to manage both virtual and physical buffers of tours,
including calculating target buffer size and determining how many tours
to release in each iteration.
"""

from typing import Dict, Tuple
import math
import numpy as np
import logging

class TourBuffer:
    """Manager for tour buffer calculations and release decisions."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the tour buffer manager.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing tour allocation parameters
        logger : logging.Logger
            Logger instance for output
        """
        self.config = config
        self.logger = logger
        
        # Extract buffer parameters from config
        self.max_pickers = config['tour_allocation']['max_pickers']
        self.avg_cycle_time = config['tour_allocation']['avg_cycle_time']
        self.prep_time = config['tour_allocation']['avg_time_to_prepare_tour']
        self.variability_factor = config['tour_allocation']['buffer_variability_factor']
        
        # Initialize buffer states
        self.unassigned_tours = {}  # Dict[tour_id, tour_data]
        self.physical_buffer = {}  # Dict[tour_id, buffer_spot_id]
        
        # Calculate target buffer size once
        self.target_buffer = self._calculate_target_buffer()
        
    def add_tours_to_pool(self, result) -> None:
        """
        Add new tours to the unassigned pool (virtual buffer).
        
        Parameters
        ----------
        result : TourFormationResult
            Result containing tour assignments and metrics
        """
        if result and result.container_assignments:
            # Extract tour data including metrics and assignments
            new_tours = {}
            for container_id, container_data in result.container_assignments.items():
                if 'tour' in container_data:
                    tour_id = container_data['tour']
                    if tour_id not in new_tours:
                        new_tours[tour_id] = {
                            'containers': [],
                            'picks': [],
                            'metrics': {}
                        }
                    new_tours[tour_id]['containers'].append(container_id)
                    
                    # Add pick assignments if available
                    if container_id in result.pick_assignments:
                        new_tours[tour_id]['picks'].extend(result.pick_assignments[container_id])
                        
            # Add metrics for each tour
            for tour_id in new_tours:
                if tour_id in result.aisle_ranges:
                    new_tours[tour_id]['aisle_range'] = result.aisle_ranges[tour_id]
                if result.metrics:
                    new_tours[tour_id]['metrics'].update(result.metrics)
            
            # Add to unassigned pool
            self.unassigned_tours.update(new_tours)
            
            self.logger.info(f"Added {len(new_tours)} tours to unassigned pool. Current pool size: {len(self.unassigned_tours)}")
            
    def get_tours_for_allocation(self, tours_to_release: int) -> Dict:
        """
        Get tours from unassigned pool for allocation to physical buffer.
        
        Parameters
        ----------
        tours_to_release : int
            Number of tours to release this iteration
            
        Returns
        -------
        Dict
            Dictionary containing selected tours and their data
        """
        # Select tours based on priority/metrics
        selected_tours = {}
        tour_ids = list(self.unassigned_tours.keys())
        
        # Take up to tours_to_release tours
        for i in range(len(tour_ids)):
            tour_id = tour_ids[i]
            selected_tours[tour_id] = self.unassigned_tours[tour_id]
            
        return selected_tours
            
    def update_physical_buffer(self, allocation_result) -> None:
        """
        Update physical buffer state based on allocation results.
        
        Parameters
        ----------
        allocation_result : TourAllocationResult
            Result containing buffer assignments
        """
        if allocation_result and allocation_result.buffer_assignments:
            # Update physical buffer state
            self.physical_buffer.update(allocation_result.buffer_assignments)
            
            # Remove allocated tours from unassigned pool
            allocated_tours = set(allocation_result.buffer_assignments.keys())
            for tour_id in allocated_tours:
                self.unassigned_tours.pop(tour_id, None)
            
            self.logger.info(f"Allocated {len(allocated_tours)} tours to physical buffer.")
            self.logger.info(f"Remaining unassigned tours: {len(self.unassigned_tours)}")
            self.logger.info(f"Current physical buffer size: {len(self.physical_buffer)}")
            
    def _calculate_target_buffer(self) -> int:
        """
        Calculate target physical buffer size.
        
        Returns
        -------
        int
            Target buffer size
        """
        # Calculate estimated buffer
        estimated_buffer = math.ceil((self.max_pickers / self.avg_cycle_time) * self.prep_time)
        
        # Apply variability factor and round up
        target_buffer = math.ceil(estimated_buffer * self.variability_factor)
        
        return target_buffer
        
    def _sample_current_buffer(self) -> int:
        """
        Sample current physical buffer size using normal distribution - assumed.
        This should come from real-world data.
        
        Returns
        -------
        int
            Sampled current buffer size
        """
        # Use mean at 50% of target buffer
        mean = self.target_buffer / 2
        # Set standard deviation to allow good spread but mostly within bounds
        std = self.target_buffer / 4
        
        # Sample until we get a value in [0, target_buffer]
        while True:
            sample = np.random.normal(mean, std)
            if 0 <= sample <= self.target_buffer:
                return len(self.physical_buffer)  # Return actual physical buffer size
                
    def calculate_tours_to_release(self) -> Tuple[int, int]:
        """
        Calculate number of tours to release in current iteration.
        
        Returns
        -------
        Tuple[int, int]
            Current buffer size and number of tours to release
        """
        # Get current physical buffer size
        current_buffer = len(self.physical_buffer)
        
        # Calculate tours to release
        tours_to_release = self.target_buffer - current_buffer
        
        self.logger.info(f"Current physical buffer size: {current_buffer}")
        self.logger.info(f"Target buffer size: {self.target_buffer}")
        self.logger.info(f"Tours to release this iteration: {max(0, tours_to_release)}")
        
        return current_buffer, max(0, tours_to_release) 