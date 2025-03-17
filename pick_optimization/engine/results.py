"""
Module for managing simulation results from pick optimization.

This module provides classes for storing and managing results from tour formation
and allocation processes during pick optimization simulation.
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from tour_allocation.ta_solver import TourAllocationResult

class SimulationResults:
    """Container for simulation results from pick optimization."""
    
    def __init__(self):
        """Initialize empty result containers."""
        # Tour Formation Results
        self.container_assignments: List[pd.DataFrame] = []  # ContainerID, TourID, etc
        self.pick_assignments: List[pd.DataFrame] = []  # TourID, SKU, Aisle, Quantity
        self.tour_metrics: List[pd.DataFrame] = []  # TourID, NumContainers, NumSKUs, etc
        self.formation_stats: List[pd.DataFrame] = []  # Overall formation stats
        
        # Tour Allocation Results
        self.buffer_assignments: List[pd.DataFrame] = []  # TourID, BufferSpotID
        self.allocation_stats: List[pd.DataFrame] = []  # Overall allocation stats
        
    def add_formation_results(self,
                            container_assignments: Dict[str, Dict],
                            pick_assignments: Dict[str, List[Dict]],
                            aisle_ranges: Dict[int, Dict],
                            iteration: int,
                            timestamp: datetime) -> None:
        """
        Add results from a tour formation iteration.
        
        Parameters
        ----------
        container_assignments : Dict[str, Dict]
            Container to tour assignments
        pick_assignments : Dict[str, List[Dict]]
            Container pick assignments
        aisle_ranges : Dict[int, Dict]
            Tour aisle ranges
        iteration : int
            Current iteration number
        timestamp : datetime
            Current simulation timestamp
        """
        # Convert container assignments to DataFrame
        container_data = []
        for container_id, info in container_assignments.items():
            container_data.append({
                'ContainerID': container_id,
                'TourID': info['tour'],
                'Iteration': iteration,
                'Timestamp': timestamp
            })
        
        # Create and store container assignments DataFrame
        if container_data:
            container_df = pd.DataFrame(container_data)
            self.container_assignments.append(container_df)
        
        # Convert pick assignments to DataFrame
        pick_data = []
        for container_id, picks in pick_assignments.items():
            for pick in picks:
                pick_row = {
                    'ContainerID': container_id,
                    'TourID': container_assignments.get(container_id, {}).get('tour'),
                    'Iteration': iteration,
                    'Timestamp': timestamp
                }
                pick_row.update(pick)  # Add all pick details
                pick_data.append(pick_row)
        
        # Create and store pick assignments DataFrame
        if pick_data:
            pick_df = pd.DataFrame(pick_data)
            self.pick_assignments.append(pick_df)
        
        # Calculate total units from pick assignments
        total_units = sum(
            pick['quantity'] 
            for container_picks in pick_assignments.values() 
            for pick in container_picks
        )
        
        # Calculate metrics per tour
        tour_metrics_list = []
        for tour_id, aisle_range in aisle_ranges.items():
            # Get containers for this tour
            tour_containers = [
                c for c, info in container_assignments.items()
                if info['tour'] == tour_id
            ]
            
            # Get picks for these containers
            tour_picks = []
            for container in tour_containers:
                if container in pick_assignments:
                    tour_picks.extend(pick_assignments[container])
            
            # Calculate distinct aisles
            distinct_aisles = len(set(pick['aisle'] for pick in tour_picks))
            
            # Calculate aisle span
            aisle_span = aisle_range['max_aisle'] - aisle_range['min_aisle']
            
            # Create tour metrics record
            tour_metrics_list.append({
                'TourID': tour_id,
                'Iteration': iteration,
                'Timestamp': timestamp,
                'ContainerCount': len(tour_containers),
                'PickCount': len(tour_picks),
                'UnitCount': sum(pick['quantity'] for pick in tour_picks),
                'DistinctAisles': distinct_aisles,
                'AisleSpan': aisle_span
            })
        
        # Create and store tour metrics DataFrame
        if tour_metrics_list:
            tour_metrics_df = pd.DataFrame(tour_metrics_list)
            self.tour_metrics.append(tour_metrics_df)
        
        # Calculate and store overall iteration stats
        total_distinct_aisles = sum(metrics['DistinctAisles'] for metrics in tour_metrics_list)
        total_aisle_span = sum(metrics['AisleSpan'] for metrics in tour_metrics_list)
        
        iteration_stats = pd.DataFrame([{
            'Iteration': iteration,
            'Timestamp': timestamp,
            'TotalTours': len(aisle_ranges),
            'TotalUnits': total_units,
            'SumDistinctAisles': total_distinct_aisles,
            'SumAisleSpan': total_aisle_span
        }])
        
        self.formation_stats.append(iteration_stats)
        
    def add_allocation_results(self,
                            allocation_result: TourAllocationResult,
                            iteration: int,
                            timestamp: datetime) -> None:
        """
        Add results from a tour allocation iteration.
        
        Parameters
        ----------
        allocation_result : TourAllocationResult
            Results from tour allocation process
        iteration : int
            Current iteration number
        timestamp : datetime
            Current simulation timestamp
        """
        # Convert tour assignments to DataFrame
        buffer_assignments_data = []
        for tour_id, buffer_id in allocation_result.buffer_assignments.items():
            buffer_assignments_data.append({
                'TourID': tour_id,
                'BufferSpotID': buffer_id,
                'ReleaseTime': timestamp
            })
        
        # Create and store buffer assignments DataFrame
        if buffer_assignments_data:
            buffer_assignments_df = pd.DataFrame(buffer_assignments_data)
            self.buffer_assignments.append(buffer_assignments_df)
        
        # Calculate allocation metrics
        total_tours = len(allocation_result.buffer_assignments)
        total_buffers = max(allocation_result.buffer_assignments.values()) + 1 if allocation_result.buffer_assignments else 0
        buffer_utilization = (total_tours / total_buffers) * 100 if total_buffers > 0 else 0
        max_concurrency = max(len(buffers) for buffers in allocation_result.aisle_assignments.values()) if allocation_result.aisle_assignments else 0
        
        # Create and store allocation stats
        allocation_stats = pd.DataFrame([{
            'Iteration': iteration,
            'Timestamp': timestamp,
            'ToursReleased': total_tours,
            'TotalBuffers': total_buffers,
            'BufferUtilization': buffer_utilization,
            'MaxAisleConcurrency': max_concurrency
        }])
        
        self.allocation_stats.append(allocation_stats)
        
    def get_consolidated_results(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get consolidated results across all iterations.
        
        Returns
        -------
        Dict[str, Optional[pd.DataFrame]]
            Dictionary containing consolidated DataFrames for each result type
        """
        return {
            'container_assignments': pd.concat(self.container_assignments, ignore_index=True)
                if self.container_assignments else None,
            'pick_assignments': pd.concat(self.pick_assignments, ignore_index=True)
                if self.pick_assignments else None,
            'tour_metrics': pd.concat(self.tour_metrics, ignore_index=True)
                if self.tour_metrics else None,
            'formation_stats': pd.concat(self.formation_stats, ignore_index=True)
                if self.formation_stats else None,
            'buffer_assignments': pd.concat(self.buffer_assignments, ignore_index=True)
                if self.buffer_assignments else None,
            'allocation_stats': pd.concat(self.allocation_stats, ignore_index=True)
                if self.allocation_stats else None
        } 