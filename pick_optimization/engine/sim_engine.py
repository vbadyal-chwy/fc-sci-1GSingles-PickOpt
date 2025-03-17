"""
Core simulation engine for pick optimization.

This module provides the main simulation engine that orchestrates the tour formation
and allocation subproblems over different time intervals.
"""

from typing import Dict, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from tabulate import tabulate

from .results import SimulationResults
from .slack_calculator import SlackCalculator
from .tour_buffer import TourBuffer
from tour_formation import run_tour_formation
from tour_allocation import run_tour_allocation

class SimEngine:
    """Core simulation engine with separate intervals for tour formation and allocation."""
    
    def __init__(self, config: Dict):
        """
        Initialize the simulation engine.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.logger = logging.getLogger('simulation_engine')
        
        # Initialize components
        self.results = SimulationResults()
        self.slack_calculator = SlackCalculator(config, self.logger)
        self.tour_buffer = TourBuffer(config, self.logger)
        
        # Get interval parameters
        self.formation_minutes = float(config['tour_formation']['tf_interval'])
        self.allocation_minutes = float(config['tour_allocation']['ta_interval'])
        
        # Initialize state
        self.containers_df = None 
        self.slotbook_data = None
        
        # Create output directories
        os.makedirs('./output/iterations', exist_ok=True)
        
    def run(self, 
            container_data: pd.DataFrame, 
            slotbook_data: pd.DataFrame,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None) -> SimulationResults:
        """
        Run simulation with dual intervals for tour formation and allocation.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
        start_time : Optional[datetime]
            Simulation start time
        end_time : Optional[datetime]
            Simulation end time
            
        Returns
        -------
        SimulationResults
            Results from the simulation run
        """
        try:
            # Initialize times
            start_time = pd.to_datetime(start_time or self.config['global']['start_time'])
            end_time = pd.to_datetime(end_time or self.config['global']['end_time'])
            
            self.logger.info(f"Starting simulation from {start_time} to {end_time}")
            
            # Initialize state
            self.containers_df = container_data.copy()
            self.containers_df['released_flag'] = False
            self.slotbook_data = slotbook_data.copy()
            
            # Calculate next event times
            next_formation = start_time
            next_allocation = start_time
            formation_delta = timedelta(minutes=self.formation_minutes)
            allocation_delta = timedelta(minutes=self.allocation_minutes)
            
            formation_iteration = 1
            allocation_iteration = 1
            
            while next_formation <= end_time or next_allocation <= end_time:
                # Determine next event
                if next_formation <= next_allocation:
                    # Run tour formation
                    self._run_tour_formation(
                        next_formation,
                        formation_iteration
                    )
                    next_formation += formation_delta
                    formation_iteration += 1
                else:
                    # Run tour allocation
                    self._run_tour_allocation(
                        next_allocation,
                        allocation_iteration
                    )
                    next_allocation += allocation_delta
                    allocation_iteration += 1
            
            # Save final results
            self._save_results(self.results.get_consolidated_results())
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise
            
    def _run_tour_formation(self, 
                           current_time: datetime,
                           iteration: int) -> None:
        """
        Run a tour formation iteration.
        
        Parameters
        ----------
        current_time : datetime
            Current simulation timestamp
        iteration : int
            Current iteration number
        """
        self.logger.info("\n########################################################################################")
        self.logger.info(f"Tour Formation Iteration {iteration} at {current_time}")
        
        # Get available containers
        available_containers = self._get_available_containers(current_time)
        
        if len(available_containers) == 0:
            self.logger.info("No containers available for tour formation")
            return
        
        # Calculate slack time for available containers
        available_containers_with_slack = self.slack_calculator.calculate_container_slack(
            available_containers, 
            current_time,
            self.slotbook_data
        )
        
        # Save available containers for debugging
        timestamp = current_time.strftime('%Y%m%d_%H%M')
        output_path = f'./output/iterations/available_containers_{timestamp}.csv'
        available_containers_with_slack.to_csv(output_path, index=False)
        
        # Run tour formation
        results = run_tour_formation(
            containers_df=available_containers_with_slack,
            skus_df=self.slotbook_data,
            config=self.config,
            logger=self.logger,
            planning_timestamp=current_time
        )
        
        if results:
            # Process results
            for result in results:
                # Add to tour buffer
                self.tour_buffer.add_tours_to_pool(result)
                
                self._update_container_tours(result.container_assignments)
                 
                # Add to results
                self.results.add_formation_results(
                    result.container_assignments,
                    result.pick_assignments,
                    result.aisle_ranges,
                    iteration,
                    current_time
                )
        else:
            self.logger.warning("No solutions found in tour formation iteration")
            
    def _run_tour_allocation(self,
                            current_time: datetime,
                            iteration: int) -> None:
        """
        Run a tour allocation iteration.
        
        Parameters
        ----------
        current_time : datetime
            Current simulation timestamp
        iteration : int
            Current iteration number
        """
        self.logger.info("\n########################################################################################")
        self.logger.info(f"Tour Allocation Iteration {iteration} at {current_time}")
        
        # Calculate number of tours to release this iteration
        current_buffer, tours_to_release = self.tour_buffer.calculate_tours_to_release()
        
        self.logger.info(f"Current buffer: {current_buffer}")
        self.logger.info(f"Tours to release: {tours_to_release}")
        
        # Skip solving if no tours to release
        if tours_to_release <= 0:
            self.logger.info("Skipping tour allocation as tours_to_release is zero")
            return
        
        # Get tours for allocation from unassigned pool
        unassigned_tours = self.tour_buffer.get_tours_for_allocation(tours_to_release)
        
        if not unassigned_tours:
            self.logger.info("No unassigned tours available for allocation")
            return
            
        # Run tour allocation
        allocation_result = run_tour_allocation(
            unassigned_tours=unassigned_tours,
            tours_to_release=tours_to_release,
            config=self.config,
            logger=self.logger
        )
        
        if allocation_result:
            # Process allocation solution
            self.results.add_allocation_results(
                allocation_result,
                iteration,
                current_time
            )
            
            # Update tour buffer state
            self.tour_buffer.update_physical_buffer(allocation_result)
            
            # Mark containers as released
            self._update_container_status(allocation_result, current_time)
        else:
            self.logger.warning("No solution found in tour allocation")
    
    def _get_available_containers(self, current_time: datetime) -> pd.DataFrame:
        """
        Get containers available for tour formation.
        
        Parameters
        ----------
        current_time : datetime
            Current simulation timestamp
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing available containers
        """
        # Filter containers that:
        # 1. Have arrived at FC (arrival_time <= current_time)
        # 2. Have not been released yet (released_flag == False)
        available = self.containers_df[
            (self.containers_df['arrive_datetime'] <= current_time) &
            (~self.containers_df['released_flag'])
        ].copy()
        
        # Filter to top 1000 unique container_ids when sorted by container_id  Todo - Remove this
        top_container_ids = sorted(available['container_id'].unique())[:100]        
        available = available[available['container_id'].isin(top_container_ids)]
        
        # For testing: filter to only the nearest multiple of max_containers_per_tour - Todo - Remove this
        num_containers = len(available['container_id'].unique())
        nearest_multiple_of_20 = (num_containers // self.config['tour_formation']['max_containers_per_tour']) * self.config['tour_formation']['max_containers_per_tour']
        available = available[available['container_id'].isin(available['container_id'].unique()[:nearest_multiple_of_20])]
        
        return available
    
    def _update_container_tours(self, container_assignments: Dict[str, Dict]) -> None:
        """
        Update container DataFrame with tour assignments.
        
        Parameters
        ----------
        container_assignments : Dict[str, Dict]
            Dictionary mapping container IDs to assignment info
        """
        # Create tour_id column if it doesn't exist
        if 'tour_id' not in self.containers_df.columns:
            self.containers_df['tour_id'] = None
        
        # Update tour_id for each container
        for container_id, info in container_assignments.items():
            if 'tour' in info:
                # Get indices for this container ID
                indices = self.containers_df[self.containers_df['container_id'] == container_id].index
                
                if not indices.empty:
                    # Assign tour_id to all rows for this container
                    self.containers_df.loc[indices, 'tour_id'] = info['tour']
                else:
                    self.logger.warning(f"Container {container_id} not found in containers DataFrame")
        
        # Log summary
        assigned_containers = self.containers_df['tour_id'].notna().sum()
        self.logger.info(f"Updated tour assignments for {assigned_containers} containers")
    
    def _update_container_status(self, allocation_result, current_time: datetime) -> None:
        """
        Update container status based on allocation results.
        
        Parameters
        ----------
        allocation_result : TourAllocationResult
            Results from tour allocation
        current_time : datetime
            Current simulation timestamp
        """
        if allocation_result and allocation_result.buffer_assignments:
            # Get all allocated tour IDs
            allocated_tours = set(allocation_result.buffer_assignments.keys())
            
            # Debug output to check column names
            self.logger.debug(f"Container DataFrame columns: {self.containers_df.columns.tolist()}")
            
            # Update container status
            mask = self.containers_df['tour_id'].isin(allocated_tours)
            self.containers_df.loc[mask, 'released_flag'] = True
            self.containers_df.loc[mask, 'release_time'] = current_time
            containers_released_count = self.containers_df[self.containers_df['released_flag'] == True]['container_id'].nunique()
            self.logger.info(f"Updated status for {containers_released_count} containers")
            

    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """
        Save final results to output directory.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Dictionary of result DataFrames to save
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_dir = f'./output/results/{self.config["global"]["wh_id"]}/{timestamp}'
            os.makedirs(output_dir, exist_ok=True)
            
            for name, df in results.items():
                if df is not None:
                    output_path = os.path.join(output_dir, f'{name}.csv')
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved {name} results to {output_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise 