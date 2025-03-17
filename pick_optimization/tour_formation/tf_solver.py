"""
Main solver module for tour formation.

This module provides the interface for solving individual tour formation problems,
handling the optimization model and solution generation.
"""

from typing import Dict, Any, List, Optional
import logging
import time
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from .tf_data import prepare_model_data
from .tf_model import TourFormationModel

@dataclass
class TourFormationResult:
    """Container for tour formation results."""
    container_assignments: Dict[str, Dict[str, Any]]
    pick_assignments: Dict[str, List[Dict[str, Any]]]
    aisle_ranges: Dict[int, Dict[str, int]]
    metrics: Dict[str, float]
    solve_time: float
    cluster_id: str

class TourFormationSolver:
    """Main solver interface for tour formation problem."""
    
    def __init__(self, container_data: pd.DataFrame, slotbook_data: pd.DataFrame, 
                 planning_timestamp: datetime, config: Dict[str, Any], num_tours: int):
        """
        Initialize the solver.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
        planning_timestamp : datetime
            Current planning timestamp
        config : Dict[str, Any]
            Configuration dictionary
        num_tours : int
            Number of tours to form
        """
        self.container_data = container_data
        self.slotbook_data = slotbook_data
        self.planning_timestamp = planning_timestamp
        self.config = config
        self.num_tours = num_tours
        
        # Extract parameters from config
        self.hourly_container_target = config['global']['hourly_container_target']
        pick_config = config['tour_formation']
        self.min_containers_per_tour = pick_config['min_containers_per_tour']
        self.max_containers_per_tour = pick_config['max_containers_per_tour']
        weights = pick_config['weights']
        self.alpha = weights['lateness']
        self.beta = weights['distinct_aisles']
        self.gamma = weights['tour_count']

        self.early_termination_seconds = pick_config['early_termination_seconds']
        self.max_cluster_size = pick_config['max_cluster_size']
        self.clustering_enabled = pick_config['clustering_enabled']
        
        # Gurobi configs
        self.output_flag = pick_config['solver']['output_flag']
        self.mip_gap = pick_config['solver']['mip_gap']
        self.time_limit = pick_config['solver']['time_limit']
        
        # Initialize Gurobi parameters
        gurobi_config = config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "GURO_PAR_ISVNAME": gurobi_config.get('ISV_NAME'),
            "GURO_PAR_ISVAPPNAME": gurobi_config.get('APP_NAME'),
            "GURO_PAR_ISVEXPIRATION": gurobi_config.get('EXPIRATION'),
            "GURO_PAR_ISVKEY": gurobi_config.get('CODE')
        }
        
        self.logger = logging.getLogger('tour_formation_solver')
        self.model_data = None
        self.model = None
        self.solution = None
        
    def prepare_data(self, container_ids=None) -> None:
        """
        Prepare data structures for optimization model.
        
        Parameters
        ----------
        container_ids : Optional[List[str]]
            Optional list of specific container IDs to include in the optimization
        """
        self.logger.info("Preparing data for optimization model...")
        
        try:
            # Prepare model data using the specified number of tours
            self.model_data = prepare_model_data(
                container_data=self.container_data,
                slotbook_data=self.slotbook_data,
                container_ids=container_ids,
                num_tours=self.num_tours,
                logger=self.logger
            )
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
        
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Solve a single tour formation problem.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Solution if found, None otherwise
        """
        try:
            start_time = time.time()
            
            # Build and solve model
            model = TourFormationModel(self.config, self.model_data, self.logger)
            model.build_model()
            solution = model.solve()
            
            if solution:
                solution['solve_time'] = time.time() - start_time
                return solution
            else:
                self.logger.warning("No feasible solution found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in tour formation solving: {str(e)}")
            raise
        
    def _adjust_tour_ids(self, solution: Dict[str, Any], offset: int) -> Dict[str, Any]:
        """
        Adjust tour IDs in solution by adding offset.
        
        Parameters
        ----------
        solution : Dict[str, Any]
            Original solution dictionary
        offset : int
            Offset to add to tour IDs
            
        Returns
        -------
        Dict[str, Any]
            Solution with adjusted tour IDs
        """
        # Adjust aisle ranges
        new_aisle_ranges = {}
        for tour_id, ranges in solution['aisle_ranges'].items():
            new_aisle_ranges[tour_id + offset] = ranges
        solution['aisle_ranges'] = new_aisle_ranges
        
        # Adjust container assignments
        for container_data in solution['container_assignments'].values():
            if 'tour_id' in container_data:
                container_data['tour_id'] += offset
                
        # Adjust pick assignments
        for pick_data in solution['pick_assignments']:
            if 'tour_id' in pick_data:
                pick_data['tour_id'] += offset
                
        return solution
        
    def _log_solution_summary(self, result: TourFormationResult) -> None:
        """Log summary of solution results."""
        self.logger.info("\nTour Formation Solution Summary:")
        self.logger.info(f"Containers assigned: {len(result.container_assignments)}")
        self.logger.info(f"Total picks: {sum(len(picks) for picks in result.pick_assignments)}")
        self.logger.info(f"Number of tours: {len(result.aisle_ranges)}")
        self.logger.info(f"Solve time: {result.solve_time:.2f} seconds")
        
        # Log metrics
        self.logger.info("\nMetrics:")
        for metric_name, metric_value in result.metrics.items():
            self.logger.info(f"{metric_name}: {metric_value:.2f}") 