import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os
from utility.logging_config import setup_logging
from tour_formation.tour_formation import TourFormationSolver                          
from tour_allocation.tour_allocation import TourAllocationSolver
import numpy as np
import math
from tabulate import tabulate
from scipy.stats import betaprime

class SimResults:
    """Container for simulation results"""
    def __init__(self):
        self.container_assignments: List[pd.DataFrame] = []
        self.pick_assignments: List[pd.DataFrame] = []
        self.tour_metrics: List[pd.DataFrame] = []
        self.picker_assignments: List[pd.DataFrame] = []
        self.tour_summaries: List[pd.DataFrame] = []
        
    def add_iteration_results(self, 
                            container_assigns: pd.DataFrame,
                            pick_assigns: pd.DataFrame,
                            tour_metrics: pd.DataFrame,
                            picker_assigns: pd.DataFrame):
        """Add results from a single iteration"""
        self.container_assignments.append(container_assigns)
        self.pick_assignments.append(pick_assigns)
        self.tour_metrics.append(tour_metrics)
        self.picker_assignments.append(picker_assigns)
        
    def get_consolidated_results(self) -> Dict[str, pd.DataFrame]:
        """Get consolidated results across all iterations"""
        return {
            'container_assignments': pd.concat(self.container_assignments, ignore_index=True) 
                if self.container_assignments else None,
            'pick_assignments': pd.concat(self.pick_assignments, ignore_index=True)
                if self.pick_assignments else None,
            'tour_metrics': pd.concat(self.tour_metrics, ignore_index=True)
                if self.tour_metrics else None,
            'picker_assignments': pd.concat(self.picker_assignments, ignore_index=True)
                if self.picker_assignments else None,
            'tour_summaries': pd.concat(self.tour_summaries, ignore_index=True)
                if self.tour_summaries else None
        }

class UnassignedTourPool:
    """Manages pool of unassigned tours waiting for allocation"""
    def __init__(self):
        self.tours: List[Dict[str, Any]] = []
        self.container_assignments: List[pd.DataFrame] = []
        self.pick_assignments: List[pd.DataFrame] = []
        self.tour_metrics: List[pd.DataFrame] = []
        
    def add_tours(self, 
                 tour_solution: Dict[str, pd.DataFrame]):
        """Add new tours to the pool"""
        self.container_assignments.append(tour_solution['container_assignments'])
        self.pick_assignments.append(tour_solution['pick_assignments'])
        self.tour_metrics.append(tour_solution['tour_metrics'])
        
    def get_consolidated_data(self) -> Dict[str, pd.DataFrame]:
        """Get consolidated data for all unassigned tours"""
        return {
            'container_assignments': pd.concat(self.container_assignments, ignore_index=True),
            'pick_assignments': pd.concat(self.pick_assignments, ignore_index=True),
            'tour_metrics': pd.concat(self.tour_metrics, ignore_index=True)
        }
        
    def clear_assigned_tours(self, assigned_tour_ids: List[int]):
        """
        Remove tours that have been allocated to pickers.
        
        Parameters
        ----------
        assigned_tour_ids : List[int]
            IDs of tours that have been assigned to pickers
        """
        # Filter out assigned tours from each component
        new_container_assigns = []
        new_pick_assigns = []
        new_tour_metrics = []
        
        for ca, pa, tm in zip(self.container_assignments, 
                            self.pick_assignments, 
                            self.tour_metrics):
            # Filter each DataFrame separately using TourID
            new_container_assigns.append(ca[~ca['TourID'].isin(assigned_tour_ids)])
            new_pick_assigns.append(pa[~pa['TourID'].isin(assigned_tour_ids)])
            new_tour_metrics.append(tm[~tm['TourID'].isin(assigned_tour_ids)])
            
        self.container_assignments = new_container_assigns
        self.pick_assignments = new_pick_assigns
        self.tour_metrics = new_tour_metrics

class SimEngine:
    """Core simulation engine with separate intervals for tour formation and allocation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging(config, 'simulation_engine')
        self.unassigned_tours = UnassignedTourPool()
        self.containers_df = None  # Will be initialized in run()
        self.slotbook_data = None  # Will be initialized in run()
        
        # Extract key parameters
        self.formation_minutes = float(config['tour_formation']['tf_interval'])
        self.allocation_minutes = float(config['tour_allocation']['ta_interval'])
        
        # Create output directories
        os.makedirs('./output/iterations', exist_ok=True)
        
    def run(self, 
            container_data: pd.DataFrame, 
            slotbook_data: pd.DataFrame,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None) -> SimResults:
        """Run simulation with dual intervals"""
        try:
            # Initialize times
            start_time = pd.to_datetime(start_time or self.config['global']['start_time'])
            end_time = pd.to_datetime(end_time or self.config['global']['end_time'])
            
            self.logger.info(f"Starting simulation from {start_time} to {end_time}")
            
            # Initialize state
            self.containers_df = container_data.copy()
            self.containers_df['released_flag'] = False
            self.slotbook_data = slotbook_data.copy()
            results = SimResults()
            
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
                        slotbook_data,
                        next_formation,
                        formation_iteration,
                        results
                    )
                    next_formation += formation_delta
                    formation_iteration += 1
                else:
                    # Run tour allocation
                    self._run_tour_allocation(
                        next_allocation,
                        allocation_iteration,
                        results
                    )
                    next_allocation += allocation_delta
                    allocation_iteration += 1
            
            # Save final results
            self._save_results(results.get_consolidated_results())
            return results
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise
            
    def _run_tour_formation(self, slotbook_data: pd.DataFrame, current_time: datetime, 
                       iteration: int, results: SimResults) -> None:
        """Run a tour formation iteration"""
        self.logger.info(f"Tour Formation Iteration {iteration} at {current_time}")
        
        # Get available containers
        available_containers = self._get_available_containers_v3(self.containers_df, current_time)
        
        if len(available_containers) == 0:
            self.logger.info("No containers available for tour formation")
            return
        
        # Calculate slack time for available containers
        available_containers_with_slack = self.calculate_container_slack(
            available_containers, 
            current_time
        )
        
        # Check if sequential optimization is enabled
        use_clustering = self.config['tour_formation'].get('clustering_enabled', False)     #bookmark
        
        if use_clustering:
            # Run sequential tour formation with clustering
            pick_solution = TourFormationSolver.solve_sequential(
                container_data=available_containers_with_slack,
                slotbook_data=slotbook_data,
                planning_timestamp=current_time,
                config=self.config,
                generate_visuals=True
            )
            
            if pick_solution:
                # Create a temporary solver instance to generate summary
                temp_solver = TourFormationSolver(
                    container_data=available_containers_with_slack,
                    slotbook_data=slotbook_data,
                    planning_timestamp=current_time,
                    config=self.config
                )
                summary_dfs = temp_solver.generate_summary(pick_solution, available_containers)
                self.unassigned_tours.add_tours(summary_dfs)
            else:
                self.logger.warning(f"No feasible solution found for sequential tour formation iteration {iteration}")
        else:
            # Run traditional (global) tour formation
            pick_solver = TourFormationSolver(
                container_data=available_containers_with_slack,
                slotbook_data=slotbook_data,
                planning_timestamp=current_time,
                config=self.config
            )
            
            pick_solution = pick_solver.solve(sequential = False)
            if pick_solution:
                # Generate summaries
                summary_dfs = pick_solver.generate_summary(pick_solution, available_containers)
                
                # Add to unassigned tour pool
                self.unassigned_tours.add_tours(summary_dfs)
            else:
                self.logger.warning(f"No feasible solution found for tour formation iteration {iteration}")
            
    def _run_tour_allocation(self,
                            current_time: datetime,
                            iteration: int,
                            results: SimResults) -> None:
        """Run a tour allocation iteration"""
        self.logger.info(f"Tour Allocation Iteration {iteration} at {current_time}")
        
        if not self.unassigned_tours.container_assignments:
            self.logger.info("No unassigned tours available for allocation")
            return
        
        # Calculate number of tours to release this iteration
        current_buffer, tours_to_release = calculate_tours_to_release(self.config)
        self.logger.info(f"Current Buffer this iteration: {current_buffer}")
        self.logger.info(f"Target tours to release this iteration: {tours_to_release}")
        
        # Skip solving if no tours to release
        if tours_to_release <= 0:
            self.logger.info("Skipping tour allocation as tours_to_release is zero")
            return
    
        # Get consolidated tour data
        tours_data = self.unassigned_tours.get_consolidated_data()
        
        # Run tour allocation
        tour_solver = TourAllocationSolver(
            tours_data=tours_data,
            config=self.config
        )
        
        allocation_solution = tour_solver.solve(tours_to_release)
        if allocation_solution:
            # Generate summaries
            allocation_summary = tour_solver.generate_summary(allocation_solution)
            
            # Update container status and inventory
            self.containers_df = self._update_container_status(
                containers_df=self.containers_df, 
                container_assignments=tours_data['container_assignments'],
                picker_assignments=allocation_summary['picker_assignments'],
                current_time=current_time
            )

            # Update inventory for assigned tours
            assigned_tour_ids = allocation_summary['picker_assignments']['TourID'].unique()
            assigned_picks = tours_data['pick_assignments'][
                tours_data['pick_assignments']['TourID'].isin(assigned_tour_ids)
            ]
            self.slotbook_data = self._update_inventory(self.slotbook_data, assigned_picks)
            
            # Store results
            results.add_iteration_results(
                tours_data['container_assignments'],
                tours_data['pick_assignments'],
                tours_data['tour_metrics'],
                allocation_summary['picker_assignments']
            )
            
            # Remove assigned tours from pool
            assigned_tour_ids = allocation_summary['picker_assignments']['TourID'].unique()
            self.unassigned_tours.clear_assigned_tours(assigned_tour_ids)
        else:
            self.logger.warning(f"No feasible solution found for tour allocation iteration {iteration}")
    
    def _get_available_containers_v3(self, 
                            containers_df: pd.DataFrame, 
                            current_time: datetime) -> pd.DataFrame:
        """
        Get containers available for planning at current time, considering complete cut time groups.
        Takes all containers up to and including the cut time that reaches/exceeds the hourly target.
        
        Parameters
        ----------
        containers_df : pd.DataFrame
            Container data with arrive_datetime and released_flag
        current_time : datetime
            Current simulation timestamp
            
        Returns
        -------
        pd.DataFrame
            Filtered container data including complete cut time groups
        """
        
        # First filter for available containers from backlog
        available = containers_df[
            (containers_df['arrive_datetime'] <= current_time) & 
            (~containers_df['released_flag'])
        ]
        
        # Filter to top 1000 unique container_ids when sorted by container_id  To Do - Remove this
        top_container_ids = sorted(available['container_id'].unique())[:1000]            #Bookmark
        available = available[available['container_id'].isin(top_container_ids)]
        
        # For testing: filter to only the nearest multiple of max_containers_per_tour - To Do - Remove this
        num_containers = len(available['container_id'].unique())
        nearest_multiple_of_20 = (num_containers // self.config['tour_formation']['max_containers_per_tour']) * self.config['tour_formation']['max_containers_per_tour']
        available = available[available['container_id'].isin(available['container_id'].unique()[:nearest_multiple_of_20])]

        
        summary_data = [
            ["Distinct Container IDs", len(available['container_id'].unique())],
            ["Distinct SKUs", len(available['item_number'].unique())],
            ["Distinct Aisles", len(available['pick_aisle'].unique())],
            ["Earliest Arrival Date", available['arrive_datetime'].min().strftime('%Y-%m-%d %H:%M')],
            ["Latest Arrival Date", available['arrive_datetime'].max().strftime('%Y-%m-%d %H:%M')],
            ["Earliest Cut Date", available['cut_datetime'].min().strftime('%Y-%m-%d %H:%M')],
            ["Latest Cut Date", available['cut_datetime'].max().strftime('%Y-%m-%d %H:%M')]
        ]
        
        self.logger.info("\nAvailable Containers Summary:")
        self.logger.info("\n" + tabulate(summary_data, headers=['Metric', 'Value'], tablefmt='grid'))
        
        return available
            
    def _get_available_containers_v1(self, 
                            containers_df: pd.DataFrame, 
                            current_time: datetime) -> pd.DataFrame:
        """
        Get containers available for planning at current time, considering complete cut time groups.
        Takes all containers up to and including the cut time that reaches/exceeds the hourly target.
        
        Parameters
        ----------
        containers_df : pd.DataFrame
            Container data with arrive_datetime and released_flag
        current_time : datetime
            Current simulation timestamp
            
        Returns
        -------
        pd.DataFrame
            Filtered container data including complete cut time groups
        """
        
        # First filter for available containers from backlog
        available = containers_df[
            (containers_df['arrive_datetime'] <= current_time) & 
            (~containers_df['released_flag'])
        ]
        
        # Get container counts by cut time
        cut_time_counts = (
            available[['container_id', 'cut_datetime']]
            .drop_duplicates()
            .groupby('cut_datetime')
            .size()
            .sort_index()
            .cumsum()
        )
        
        hourly_target = self.config['global']['hourly_container_target']
        
        # Find the first cut time that meets or exceeds the target
        target_cut_times = cut_time_counts[cut_time_counts >= hourly_target]
        if not target_cut_times.empty:
            # Include all cut times up to and including the first one that exceeds target
            last_included_cut = target_cut_times.index[0]
        else:
            # If we never reach the target, include all cut times
            last_included_cut = cut_time_counts.index[-1] if not cut_time_counts.empty else None
        
        # Get all containers up to and including the selected cut time
        if last_included_cut is not None:
            selected_cut_times = cut_time_counts.index[cut_time_counts.index <= last_included_cut]
            available = available[available['cut_datetime'].isin(selected_cut_times)]
            
            '''# For testing: filter to only the nearest multiple of 20 distinct containers - To Do - Remove this
            num_containers = len(available['container_id'].unique())
            nearest_multiple_of_20 = (num_containers // 20) * 20
            available = available[available['container_id'].isin(available['container_id'].unique()[:nearest_multiple_of_20])]
            '''
            # Create and print summary table
            summary_data = [
                ["Distinct Container IDs", len(available['container_id'].unique())],
                ["Distinct SKUs", len(available['item_number'].unique())],
                ["Distinct Aisles", len(available['pick_aisle'].unique())],
                ["Earliest Arrival Date", available['arrive_datetime'].min().strftime('%Y-%m-%d %H:%M')],
                ["Latest Arrival Date", available['arrive_datetime'].max().strftime('%Y-%m-%d %H:%M')],
                ["Earliest Cut Date", available['cut_datetime'].min().strftime('%Y-%m-%d %H:%M')],
                ["Latest Cut Date", available['cut_datetime'].max().strftime('%Y-%m-%d %H:%M')]
            ]
            
            self.logger.info("\nAvailable Containers Summary:")
            self.logger.info("\n" + tabulate(summary_data, headers=['Metric', 'Value'], tablefmt='grid'))
            
            return available

        # If no cut times found, return empty DataFrame with same structure
        return available.head(0)

    def _run_iteration(self,
                      available_containers: pd.DataFrame,
                      slotbook_data: pd.DataFrame,
                      current_time: datetime,
                      iteration: int) -> Optional[Dict[str, Any]]:
        """Run a single iteration of pick planning and tour allocation"""
        try:
           
            self.logger.info("Running Tour Formation Solver") 
            
            # Initialize pick planning solver
            pick_solver = TourFormationSolver(
                container_data=available_containers,
                slotbook_data=slotbook_data,
                planning_timestamp=current_time,
                config=self.config
            )

            
            pick_solution = pick_solver.solve()
            if not pick_solution:
                return None
                
            summary_dfs = pick_solver.generate_summary(pick_solution, available_containers)
            
            
            self.logger.info("Running Tour Allocation Solver") 
            
            # Initialize tour allocation solver
            tour_solver = TourAllocationSolver(
                tours_data=summary_dfs,
                config=self.config
            )
            
            allocation_solution = tour_solver.solve()
            if not allocation_solution:
                return None
                
            allocation_summary = tour_solver.generate_summary(allocation_solution)
            
            # Combine results
            results = {**summary_dfs, **allocation_summary}
            results['tour_summary'] = self._create_tour_summary(
                summary_dfs['tour_metrics'],
                iteration,
                current_time
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {str(e)}")
            raise
            
    def _update_container_status(self,
                                containers_df: pd.DataFrame,
                                container_assignments: pd.DataFrame,
                                picker_assignments: pd.DataFrame,
                                current_time: datetime) -> pd.DataFrame:
        """
        Update container status with assignment results.
        
        Parameters
        ----------
        containers_df : pd.DataFrame
            Current container data
        container_assignments : pd.DataFrame
            Container-tour assignments
        picker_assignments : pd.DataFrame
            Picker-tour assignments
        current_time : datetime
            Current simulation time
            
        Returns
        -------
        pd.DataFrame
            Updated container data
        """
        # Create a copy to avoid modifying the original
        updated_containers = containers_df.copy()
        
        # Merge on TourID to get PickerID for each container
        released_containers = container_assignments.merge(
            picker_assignments[['TourID', 'PickerID']],
            on='TourID',
            how='left'
        )
        
        # Update container status
        for _, row in released_containers.iterrows():
            mask = updated_containers['container_id'] == row['ContainerID']
            if mask.any():
                updated_containers.loc[mask, 'released_flag'] = True
                updated_containers.loc[mask, 'tour_id'] = row['TourID']
                updated_containers.loc[mask, 'tour_release_time'] = current_time
                updated_containers.loc[mask, 'picker_id'] = row['PickerID']
            
        return updated_containers
        
        
    def _update_inventory(self,
                         slotbook_data: pd.DataFrame,
                         pick_assignments: pd.DataFrame) -> pd.DataFrame:
        """
        Update inventory levels based on picks.
        
        Parameters
        ----------
        slotbook_data : pd.DataFrame
            Current inventory data with columns ['item_number', 'location_id', 'actual_qty']
        pick_assignments : pd.DataFrame
            Pick assignments with columns ['SKU', 'Aisle', 'Quantity']
            
        Returns
        -------
        pd.DataFrame
            Updated slotbook data with decremented quantities
        """
        # First verify the dataframe has required columns
        required_cols = {'SKU', 'Aisle', 'Quantity'}
        missing_cols = required_cols - set(pick_assignments.columns)
        if missing_cols:
            raise ValueError(f"Pick assignments missing required columns: {missing_cols}")
        
        # Update inventory using aisle numbers
        for _, pick in pick_assignments.iterrows():
            mask = (
                (slotbook_data['item_number'] == pick['SKU']) & 
                (slotbook_data['aisle_sequence'] == pick['Aisle'])
            )
            if not mask.any():
                self.logger.warning(
                    f"No matching inventory found for SKU {pick['SKU']} in aisle {pick['Aisle']}"
                )
                continue
                
            slotbook_data.loc[mask, 'actual_qty'] -= pick['Quantity']
                
        # Verify no negative quantities
        neg_qty_mask = slotbook_data['actual_qty'] < 0
        if neg_qty_mask.any():
            neg_locations = slotbook_data[neg_qty_mask]
            self.logger.warning(
                f"Found {len(neg_locations)} locations with negative quantities after update"
            )
                
        return slotbook_data
        
    def _create_tour_summary(self,
                        tour_metrics: pd.DataFrame,
                        iteration: int,
                        current_time: datetime) -> pd.DataFrame:
        """Create summary for current iteration's tours"""
        summary = tour_metrics.copy()
        summary['Iteration'] = iteration
        summary['PlanningTimestamp'] = current_time
        summary['WarehouseID'] = self.config['global']['wh_id']
        return summary
        
    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Save final results to output directory"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'../output/{self.config["global"]["wh_id"]}/{timestamp}'
            os.makedirs(output_dir, exist_ok=True)
            
            for name, df in results.items():
                if df is not None:
                    output_path = os.path.join(output_dir, f'{name}.csv')
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved {name} results to {output_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _check_tour_formation_feasibility(self, 
                                   container_data: pd.DataFrame, 
                                   slotbook_data: pd.DataFrame) -> bool:
        """
        Check feasibility of tour formation problem.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
            
        Returns
        -------
        bool
            True if the problem is feasible, False otherwise
        """
        try:
            self.logger.info("Checking tour formation feasibility...")
            
            # Extract configuration parameters
            pick_config = self.config['tour_formation']
            min_containers = pick_config['min_containers_per_tour']
            max_containers = pick_config['max_containers_per_tour']

            # 1. Basic Input Data Checks
            if len(container_data) == 0:
                self.logger.error("No containers to process")
                return False
                
            # Check if all SKUs have valid aisle locations
            container_skus = set(container_data['item_number'].unique())
            slotbook_skus = set(slotbook_data['item_number'].unique())
            missing_skus = container_skus - slotbook_skus
            
            if missing_skus:
                self.logger.error(f"Found SKUs with no valid picking locations: {missing_skus}")
                return False
                
            # 2. Capacity Feasibility
            num_containers = len(container_data['container_id'].unique())
            min_tours_needed = -(-num_containers // max_containers)  # Ceiling division
            
            if min_tours_needed * max_containers < num_containers:
                self.logger.error(
                    f"Not enough tours ({num_containers}) to satisfy containers "
                    f"per tour ({max_containers}) for required tours ({min_tours_needed})"
                )
                return False
                
            # 3. Inventory Feasibility
            for sku in container_skus:
                total_required = container_data[
                    container_data['item_number'] == sku
                ]['quantity'].sum()
                
                total_available = slotbook_data[
                    slotbook_data['item_number'] == sku
                ]['actual_qty'].sum()
                
                if total_required > total_available:
                    self.logger.error(
                        f"Insufficient inventory for SKU {sku}. "
                        f"Required: {total_required}, Available: {total_available}"
                    )
                    return False
                    
            self.logger.info("Tour formation feasibility checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in tour formation feasibility check: {str(e)}")
            return False

    def _check_tour_allocation_feasibility(self, 
                                        container_data: pd.DataFrame) -> bool:
        """
        Check feasibility of tour allocation optimization problem.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
            
        Returns
        -------
        bool
            True if the problem is feasible, False otherwise
        """
        try:
            self.logger.info("Checking tour allocation feasibility...")
            
            # Extract configuration parameters
            max_pickers = self.config['tour_allocation']['max_pickers']
            min_containers = self.config['tour_formation']['min_containers_per_tour']
            max_containers = self.config['tour_formation']['max_containers_per_tour']
            
            # 1. Basic Resource Checks
            if max_pickers <= 0:
                self.logger.error("No pickers available for allocation")
                return False
            
            # 2. Tour-Picker Ratio Feasibility
            num_containers = len(container_data['container_id'].unique())
            min_tours = -(-num_containers // max_containers)  # Ceiling division
            
            if min_tours < max_pickers:
                self.logger.warning(
                    f"Fewer required tours ({min_tours}) than available pickers ({max_pickers})"
                )   
            
            self.logger.info("Tour allocation feasibility checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in tour allocation feasibility check: {str(e)}")
            return False
    
    def calculate_container_slack(self, 
                          container_data: pd.DataFrame, 
                          current_time: datetime) -> pd.DataFrame:
        """
        Calculate the slack time for each container, representing the buffer time 
        available before the container will miss its critical pull time.
        
        Slack = CPTi - (Pi + Ti + OTi + BrTi + Wi)
        
        Where:
        - CPTi: Critical Pull Time for container i
        - Pi: Picking time for container i based on SKUs and quantities
        - Ti: Travel time based on aisles visited
        - OTi: Other buffer time for grooming, packing, shipping (constant)
        - BrTi: Break time impact based on time of day capacity
        - Wi: Waiting time (WQi + WBi) including queue and buffer wait
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information with container_id, item_number, 
            quantity, pull_datetime
        current_time : datetime
            Current simulation timestamp
            
        Returns
        -------
        pd.DataFrame
            Original container_data with added 'slack' column in minutes
        """
        try:
            self.logger.info("Calculating slack time for containers...")
            
            # Create a copy to avoid modifying the original
            result_df = container_data.copy()
            
            # Constants
            OTHER_TIME_BUFFER = 30  # minutes for grooming, packing, shipping
            TRAVEL_TIME_CONSTANT = np.random.normal(5, 3)  # minutes per aisle        
            AISLE_CROSSING_FACTOR = 1/8  # factor for crossing aisles vs traveling in aisles
            
            # Create a time-of-day capacity dictionary (0=no capacity, 1=full capacity)
            # This represents shift changes, breaks, lunches, etc.
            capacity_by_hour = {
                0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0,  # Nightshift (10:00 PM - 1:30 AM, 2:00 AM - 5:00 AM)
                6: 0.0, 7: 0.5, 8: 1.0, 9: 1.0, 10: 0.5,  # Dayshift (7:30 AM - 10:30 AM)
                11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 0.5,  # Dayshift (11:00 AM - 3:00 PM)
                16: 1.0, 17: 1.0, 18: 0.5,  # Dayshift (3:30 PM - 6:00 PM)
                19: 1.0, 20: 1.0, 21: 1.0, 22: 0.5, 23: 1.0  # Nightshift (6:30 PM - 9:30 PM, 10:00 PM - 1:30 AM)
            }
            
            # Parameters for Pearson Type VI distribution (from time studies)
            gamma = 0.065696   # location
            beta = 10.431830   # scale
            alpha1 = 8.982293  # shape parameter a
            alpha2 = 4.289152  # shape parameter b
            
            # Create the betaprime distribution
            dist = betaprime(a=alpha1, b=alpha2, scale=beta)
            
            # Create a random number generator with seed for reproducibility
            rng = np.random.RandomState(42)
            
            # Calculate configuration parameters once
            hourly_container_target = self.config['global']['hourly_container_target']
            active_pickers = self.config['tour_allocation']['max_pickers']
            avg_cycle_time = self.config['tour_allocation']['avg_cycle_time']  # in minutes
            buffer_wait_minutes = (active_pickers / avg_cycle_time) * self.config['tour_allocation']['avg_time_to_prepare_tour']
            
            # Count containers with same or earlier pull_datetime once (for queue wait time)
            pull_datetime_counts = container_data.groupby('pull_datetime')['container_id'].nunique().cumsum()
            
            # Precompute time components
            result_df['time_until_pull'] = 0.0
            result_df['picking_time'] = 0.0
            result_df['travel_time'] = 0.0
            result_df['other_time'] = OTHER_TIME_BUFFER  # Constant for all containers
            result_df['break_impact'] = 0.0
            result_df['waiting_time'] = 0.0
            result_df['slack_minutes'] = 0.0
            
            # Create lookup tables for SKU aisles and inventory levels
            sku_aisle_lookup = {}
            sku_inventory_lookup = {}
            
            for sku in result_df['item_number'].unique():
                sku_data = self.slotbook_data[self.slotbook_data['item_number'] == sku]
                sku_aisles = sku_data['aisle_sequence'].unique()
                sku_aisle_lookup[sku] = sku_aisles
                
                # Create inventory lookup for each SKU-aisle combination
                for aisle in sku_aisles:
                    inventory = sku_data[sku_data['aisle_sequence'] == aisle]['actual_qty'].sum()
                    sku_inventory_lookup[(sku, aisle)] = inventory
            
            # Group by container to process each one
            container_groups = result_df.groupby('container_id')
            
            for container_id, container_items in container_groups:
                pull_datetime = container_items['pull_datetime'].iloc[0]
                container_mask = result_df['container_id'] == container_id
                
                # 1. Calculate time until pull
                time_until_pull = max(0, (pull_datetime - current_time).total_seconds() / 60)
                result_df.loc[container_mask, 'time_until_pull'] = time_until_pull
                
                # 2. Calculate picking time
                picking_time_minutes = 0
                for _, row in container_items.iterrows():
                    # Sample from Pearson VI distribution for this SKU
                    sku_base_time = gamma + dist.rvs(random_state=rng)
                    # Adjust for quantity: roundup(quantity / 3)
                    quantity_factor = math.ceil(row['quantity'] / 3)
                    picking_time_minutes += sku_base_time * quantity_factor
                
                # Convert to minutes
                picking_time_minutes /= 60
                result_df.loc[container_mask, 'picking_time'] = picking_time_minutes
                
                # 3. Calculate travel time - find best aisles for each SKU in container
                container_aisles = set()
                for _, row in container_items.iterrows():
                    sku = row['item_number']
                    sku_aisles = sku_aisle_lookup.get(sku, [])
                    
                    if len(sku_aisles) == 1:
                        container_aisles.add(sku_aisles[0])
                    elif len(sku_aisles) > 1:
                        # Find aisle with highest inventory
                        best_aisle = max(
                            [(aisle, sku_inventory_lookup.get((sku, aisle), 0)) 
                            for aisle in sku_aisles],
                            key=lambda x: x[1]
                        )[0]
                        container_aisles.add(best_aisle)
                
                # Calculate travel time
                if container_aisles:
                    aisles_in = len(container_aisles)
                    aisles_across = max(container_aisles) - min(container_aisles) if aisles_in > 1 else 0
                    travel_time_minutes = (aisles_in + aisles_across * AISLE_CROSSING_FACTOR) * TRAVEL_TIME_CONSTANT
                else:
                    travel_time_minutes = 0
                    
                result_df.loc[container_mask, 'travel_time'] = travel_time_minutes
                
                # 4. Calculate break impact time
                if pull_datetime > current_time:
                    break_time_minutes = 0
                    current_hour = current_time.hour
                    pull_hour = pull_datetime.hour
                    
                    # Efficiently calculate break impact
                    hour_ptr = current_hour
                    while hour_ptr != pull_hour:
                        capacity_factor = capacity_by_hour[hour_ptr]
                        if capacity_factor < 1.0:
                            break_time_minutes += 60 * (1.0 - capacity_factor)
                        hour_ptr = (hour_ptr + 1) % 24
                    
                    # Check final hour
                    capacity_factor = capacity_by_hour[pull_hour]
                    if capacity_factor < 1.0:
                        break_time_minutes += 60 * (1.0 - capacity_factor)
                        
                    result_df.loc[container_mask, 'break_impact'] = break_time_minutes
                
                # 5. Calculate waiting time
                # 5a. Queue waiting time - use precomputed counts
                urgent_containers = pull_datetime_counts.get(pull_datetime, 0)
                queue_wait_minutes = (urgent_containers / hourly_container_target) * 60
                
                # 5b. Total waiting time (queue + buffer)
                waiting_time_minutes = queue_wait_minutes + buffer_wait_minutes
                result_df.loc[container_mask, 'waiting_time'] = waiting_time_minutes
                
                # 6. Calculate slack
                total_processing_time = (
                    picking_time_minutes + 
                    travel_time_minutes + 
                    OTHER_TIME_BUFFER + 
                    result_df.loc[container_mask, 'break_impact'].iloc[0] + 
                    waiting_time_minutes
                )
                
                slack_minutes = time_until_pull - total_processing_time
                result_df.loc[container_mask, 'slack_minutes'] = slack_minutes
                
                # Log detailed breakdown for this container
                self.logger.debug(
                    f"Container {container_id} slack calculation:\n"
                    f"  - Pull time: {pull_datetime}\n"
                    f"  - Time until pull: {time_until_pull:.2f} minutes\n"
                    f"  - Picking time: {picking_time_minutes:.2f} minutes\n"
                    f"  - Travel time: {travel_time_minutes:.2f} minutes\n"
                    f"  - Other time: {OTHER_TIME_BUFFER:.2f} minutes\n"
                    f"  - Break impact: {result_df.loc[container_mask, 'break_impact'].iloc[0]:.2f} minutes\n"
                    f"  - Waiting time: {waiting_time_minutes:.2f} minutes\n"
                    f"  - Total slack: {slack_minutes:.2f} minutes"
                )
            
            # Add slack category
            result_df['slack_category'] = pd.cut(
                result_df['slack_minutes'], 
                bins=[-float('inf'), 0, 60, float('inf')],
                labels=['Critical', 'Urgent', 'Safe']
            )
            
            # Log summary statistics
            # First get the slack category for each container (taking first occurrence since all rows for a container have same slack)
            container_categories = result_df.groupby('container_id')['slack_category'].first()

            # Then count containers in each category
            category_counts = container_categories.value_counts()
            
            summary_data = [
                ["Critical (slack < 0 min.)",  category_counts.get('Critical', 0)],
                ["Urgent (0 <= slack < 60 min.)", category_counts.get('Urgent', 0)], 
                ["Safe (slack >= 60 min.)", category_counts.get('Safe', 0)],
                ["Total Containers", len(container_groups)]
            ]
            
            self.logger.info("\nSlack Calculation Summary:")
            self.logger.info("\n" + tabulate(summary_data, headers=['Category', 'Count'], tablefmt='grid'))            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating container slack: {str(e)}")
            raise
            
def calculate_target_buffer(config: Dict[str, Any]) -> int:
    """
    Calculate target physical buffer size.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing tour allocation parameters
    
    Returns
    -------
    int
        Target buffer size
    """
    # Extract parameters
    max_pickers = config['tour_allocation']['max_pickers']
    avg_cycle_time = config['tour_allocation']['avg_cycle_time']
    prep_time = config['tour_allocation']['avg_time_to_prepare_tour']
    variability_factor = config['tour_allocation']['buffer_variability_factor']
    
    # Calculate estimated buffer
    estimated_buffer = math.ceil((max_pickers / avg_cycle_time) * prep_time)
    
    # Apply variability factor and round up
    target_buffer = math.ceil(estimated_buffer * variability_factor)
    
    return target_buffer

def sample_current_buffer(target_buffer: int) -> int:
    """
    Sample current physical buffer size using truncated normal distribution.
    
    Parameters
    ----------
    target_buffer : int
        Target buffer size (used as upper bound)
    
    Returns
    -------
    int
        Sampled current buffer size
    """
    # Use mean at 50% of target buffer
    mean = target_buffer / 2
    # Set standard deviation to allow good spread but mostly within bounds
    std = target_buffer / 4
    
    # Sample until we get a value in [0, target_buffer]
    while True:
        sample = np.random.normal(mean, std)
        if 0 <= sample <= target_buffer:
            #return round(sample)
            return 0

def calculate_tours_to_release(config: Dict[str, Any]) -> int:
    """
    Calculate number of tours to release in current iteration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing tour allocation parameters
    
    Returns
    -------
    int
        Number of tours to release
    """
    # Calculate target buffer
    target_buffer = calculate_target_buffer(config)
    
    # Sample current buffer
    current_buffer = sample_current_buffer(target_buffer)
    
    # Calculate tours to release
    tours_to_release = target_buffer - current_buffer
    
    return current_buffer, max(0, tours_to_release)


    
    