"""
Slack calculator module for pick optimization.

This module provides functionality to calculate slack time for containers,
i.e. the buffer time available before a container will miss its
critical pull time. Slack will be used to emphasize containers in our subproblems
"""

# Standard library imports
import logging
import math
import time
from datetime import datetime
from typing import Dict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import betaprime
from tabulate import tabulate

# Get module-specific logger with workflow logging
from logging_config import get_logger
logger = get_logger(__name__, 'tour_formation')

class SlackCalculator:
    """
    Calculator for container slack times.
    
    Calculates slack time using the formula:
    Slack = CPTi - (Pi + Ti + OTi + BrTi + Wi)
    
    Where:
    - CPTi: Critical Pull Time for container i
    - Pi: Picking time for container i based on SKUs and quantities
    - Ti: Travel time based on aisles visited
    - OTi: Other buffer time for grooming, packing, shipping (constant)
    - BrTi: Break time impact based on time of day capacity
    - Wi: Waiting time (WQi + WBi) including queue and buffer wait
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the slack calculator.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing simulation parameters
        logger : logging.Logger
            Logger instance for output
        """
        self.config = config
        self.logger = logger
        
        # Get constants from config
        self.OTHER_TIME_BUFFER = self.config['slack_calculation']['other_time_buffer']  # minutes for grooming, packing, shipping
        self.TRAVEL_TIME_CONSTANT = self.config['slack_calculation']['travel_time_constant']  # minutes per aisle
        self.AISLE_CROSSING_FACTOR = self.config['slack_calculation']['aisle_crossing_factor']  # factor for crossing aisles vs traveling in aisles
        
        # Get slack thresholds from config
        self.CRITICAL_SLACK_THRESHOLD = self.config['slack_calculation']['critical_slack_threshold']
        self.URGENT_SLACK_THRESHOLD = self.config['slack_calculation']['urgent_slack_threshold']
        
        # Get capacity by hour from config
        self.capacity_by_hour = self.config['slack_calculation']['capacity_by_hour']
        
        # Parameters for Pick Time Distribution (Pearson Type VI distribution - from time studies)
        self.gamma = 0.065696   # location
        self.beta = 10.431830   # scale
        self.alpha1 = 8.982293  # shape parameter a
        self.alpha2 = 4.289152  # shape parameter b
        
        # Create the betaprime distribution
        self.dist = betaprime(a=self.alpha1, b=self.alpha2, scale=self.beta)
        
        # Create a random number generator with seed for reproducibility
        self.rng = np.random.RandomState(42)
        
    def _create_sku_lookups(self, result_df: pd.DataFrame, slotbook_data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Create lookup tables for SKU aisles and inventory levels using optimized vectorized operations."""
        # Get unique SKUs from result_df to process only what's needed
        required_skus = result_df['item_number'].unique()
        
        # Filter slotbook to only required SKUs for efficiency
        filtered_slotbook = slotbook_data[slotbook_data['item_number'].isin(required_skus)]
        
        # Create SKU-aisle mapping using groupby (vectorized)
        sku_aisle_lookup = (
            filtered_slotbook.groupby('item_number')['aisle_sequence']
            .apply(lambda x: x.unique().tolist())
            .to_dict()
        )
        
        # Create SKU-inventory lookup using vectorized operations
        inventory_agg = (
            filtered_slotbook.groupby(['item_number', 'aisle_sequence'])['actual_qty']
            .sum()
            .reset_index()
        )
        
        sku_inventory_lookup = {
            (row['item_number'], row['aisle_sequence']): row['actual_qty']
            for _, row in inventory_agg.iterrows()
        }
                
        return sku_aisle_lookup, sku_inventory_lookup

    def _calculate_picking_times_vectorized(self, container_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate picking times for all containers using vectorized operations."""
        # Generate random values for all rows at once for efficiency
        num_rows = len(container_data)
        random_values = self.dist.rvs(size=num_rows, random_state=self.rng)
        
        # Vectorized calculation of picking times
        container_data = container_data.copy()
        container_data['sku_base_time'] = self.gamma + random_values
        container_data['quantity_factor'] = np.ceil(container_data['pick_quantity'] / 3)
        container_data['item_picking_time'] = container_data['sku_base_time'] * container_data['quantity_factor']
        
        # Sum by container
        picking_times = (
            container_data.groupby('container_id')['item_picking_time']
            .sum()
            .reset_index()
        )
        picking_times['picking_time'] = picking_times['item_picking_time'] / 60  # Convert to hours
        picking_times = picking_times[['container_id', 'picking_time']]
        
        return picking_times

    def _calculate_travel_times_vectorized(self, container_data: pd.DataFrame, sku_aisle_lookup: Dict, sku_inventory_lookup: Dict) -> pd.DataFrame:
        """Calculate travel times for all containers using vectorized operations."""
        travel_times = []
        
        for container_id, container_items in container_data.groupby('container_id'):
            container_aisles = set()
            
            for _, row in container_items.iterrows():
                sku = row['item_number']
                sku_aisles = sku_aisle_lookup.get(sku, [])
                
                if len(sku_aisles) == 1:
                    container_aisles.add(sku_aisles[0])
                elif len(sku_aisles) > 1:
                    best_aisle = max(
                        [(aisle, sku_inventory_lookup.get((sku, aisle), 0)) 
                         for aisle in sku_aisles],
                        key=lambda x: x[1]
                    )[0]
                    container_aisles.add(best_aisle)
            
            if container_aisles:
                aisles_in = len(container_aisles)
                aisles_across = max(container_aisles) - min(container_aisles) if aisles_in > 1 else 0
                travel_time = (aisles_in + aisles_across * (1.0/self.AISLE_CROSSING_FACTOR)) * self.TRAVEL_TIME_CONSTANT
            else:
                travel_time = 0
                
            travel_times.append({'container_id': container_id, 'travel_time': travel_time})
        
        return pd.DataFrame(travel_times)

    def _calculate_break_impacts_vectorized(self, container_data: pd.DataFrame, current_time: datetime) -> pd.DataFrame:
        """Calculate break impacts for all containers using vectorized operations."""
        # Get unique pull datetimes to avoid redundant calculations
        unique_pull_times = container_data['pull_datetime'].unique()
        
        # Calculate break impact for each unique pull time
        break_impact_lookup = {}
        for pull_datetime in unique_pull_times:
            break_impact_lookup[pull_datetime] = self._calculate_break_impact(current_time, pull_datetime)
        
        # Map break impacts to containers
        break_impacts = (
            container_data[['container_id', 'pull_datetime']]
            .drop_duplicates()
            .copy()
        )
        break_impacts['break_impact'] = break_impacts['pull_datetime'].map(break_impact_lookup)
        
        return break_impacts[['container_id', 'break_impact']]

    def _calculate_picking_time(self, container_items: pd.DataFrame) -> float:
        """Calculate picking time for a container."""
        picking_time_minutes = 0
        for _, row in container_items.iterrows():
            sku_base_time = self.gamma + self.dist.rvs(random_state=self.rng)
            quantity_factor = math.ceil(row['pick_quantity'] / 3)
            picking_time_minutes += sku_base_time * quantity_factor
        return picking_time_minutes / 60

    def _calculate_travel_time(self, container_items: pd.DataFrame, sku_aisle_lookup: Dict, sku_inventory_lookup: Dict) -> float:
        """Calculate travel time for a container."""
        container_aisles = set()
        for _, row in container_items.iterrows():
            sku = row['item_number']
            sku_aisles = sku_aisle_lookup.get(sku, [])
            
            if len(sku_aisles) == 1:
                container_aisles.add(sku_aisles[0])
            elif len(sku_aisles) > 1:
                best_aisle = max(
                    [(aisle, sku_inventory_lookup.get((sku, aisle), 0)) 
                     for aisle in sku_aisles],
                    key=lambda x: x[1]
                )[0]
                container_aisles.add(best_aisle)
        
        if not container_aisles:
            return 0
            
        aisles_in = len(container_aisles)
        aisles_across = max(container_aisles) - min(container_aisles) if aisles_in > 1 else 0
        return (aisles_in + aisles_across * (1.0/self.AISLE_CROSSING_FACTOR)) * self.TRAVEL_TIME_CONSTANT

    def _calculate_break_impact(self, current_time: datetime, pull_datetime: datetime) -> float:
        """Calculate break impact time."""
        if pull_datetime <= current_time:
            return 0
            
        break_time_minutes = 0
        current_hour = current_time.hour
        pull_hour = pull_datetime.hour
        
        hour_ptr = current_hour
        while hour_ptr != pull_hour:
            capacity_factor = self.capacity_by_hour[hour_ptr]
            if capacity_factor < 1.0:
                break_time_minutes += 60 * (1.0 - capacity_factor)
            hour_ptr = (hour_ptr + 1) % 24
        
        capacity_factor = self.capacity_by_hour[pull_hour]
        if capacity_factor < 1.0:
            break_time_minutes += 60 * (1.0 - capacity_factor)
            
        return break_time_minutes

    def calculate_container_slack(self, 
                                container_data: pd.DataFrame, 
                                current_time: datetime,
                                slotbook_data: pd.DataFrame,
                                labor_headcount: int,
                                container_target: int) -> pd.DataFrame:
        """Calculate the slack time for each container using optimized vectorized operations."""
        try:
            start_time = time.time()
            self.logger.info("Calculating slack time for containers...")
            
            # Initialize result DataFrame
            result_df = container_data.copy()
            result_df['other_time'] = self.OTHER_TIME_BUFFER
            
            # Phase 1: Vectorized time until pull calculation
            time_calc_start = time.time()
            result_df['time_until_pull'] = (result_df['pull_datetime'] - current_time).dt.total_seconds() / 60
            result_df['time_until_pull'] = result_df['time_until_pull'].clip(lower=0)
            time_calc_time = time.time() - time_calc_start
            
            # Phase 2: Pre-compute lookups and waiting times
            lookup_start = time.time()
            
            # Get configuration parameters
            buffer_wait_minutes = self.config['slack_calculation']['buffer_variability_factor'] * \
                self.config['slack_calculation']['avg_time_to_prepare_tour']
            
            # Create optimized lookups
            sku_aisle_lookup, sku_inventory_lookup = self._create_sku_lookups(result_df, slotbook_data)
            
            # Pre-compute waiting times using vectorized operations
            pull_datetime_counts = container_data.groupby('pull_datetime')['container_id'].nunique()
            waiting_time_lookup = {}
            for pull_datetime, urgent_containers in pull_datetime_counts.items():
                queue_wait_minutes = (urgent_containers / container_target) * 60
                waiting_time_lookup[pull_datetime] = queue_wait_minutes + buffer_wait_minutes
            
            lookup_time = time.time() - lookup_start
            
            # Phase 3: Vectorized component calculations
            components_start = time.time()
            
            # Calculate picking times for all containers
            picking_times_df = self._calculate_picking_times_vectorized(container_data)
            
            # Calculate travel times for all containers
            travel_times_df = self._calculate_travel_times_vectorized(container_data, sku_aisle_lookup, sku_inventory_lookup)
            
            # Calculate break impacts for all containers
            break_impacts_df = self._calculate_break_impacts_vectorized(container_data, current_time)
            
            components_time = time.time() - components_start
            
            # Phase 4: Merge all components efficiently
            merge_start = time.time()
            
            # Get container-level data (one row per container)
            container_level = result_df[['container_id', 'pull_datetime']].drop_duplicates()
            
            # Add waiting times
            container_level['waiting_time'] = container_level['pull_datetime'].map(waiting_time_lookup)
            
            # Merge all time components
            container_level = container_level.merge(picking_times_df, on='container_id', how='left')
            container_level = container_level.merge(travel_times_df, on='container_id', how='left')
            container_level = container_level.merge(break_impacts_df, on='container_id', how='left')
            
            # Fill any missing values with 0
            time_columns = ['picking_time', 'travel_time', 'break_impact', 'waiting_time']
            for col in time_columns:
                container_level[col] = container_level[col].fillna(0)
            
            merge_time = time.time() - merge_start
            
            # Phase 5: Vectorized slack calculation
            slack_calc_start = time.time()
            
            # Calculate total processing time vectorized
            container_level['total_processing_time'] = (
                container_level['picking_time'] + 
                container_level['travel_time'] + 
                self.OTHER_TIME_BUFFER + 
                container_level['break_impact'] + 
                container_level['waiting_time']
            )
            
            # Get time until pull for containers
            container_time_until_pull = result_df[['container_id', 'time_until_pull']].drop_duplicates()
            container_level = container_level.merge(container_time_until_pull, on='container_id', how='left')
            
            # Calculate slack vectorized
            container_level['slack_minutes'] = container_level['time_until_pull'] - container_level['total_processing_time']
            
            slack_calc_time = time.time() - slack_calc_start
            
            # Phase 6: Merge back to original DataFrame
            final_merge_start = time.time()
            
            # Merge container-level calculations back to result_df
            merge_columns = ['container_id', 'waiting_time', 'picking_time', 'travel_time', 'break_impact', 'slack_minutes']
            result_df = result_df.merge(
                container_level[merge_columns], 
                on='container_id', 
                how='left'
            )
            
            final_merge_time = time.time() - final_merge_start
            
            # Phase 7: Slack categorization and priority handling
            categorization_start = time.time()
            
            # Add slack categories using config thresholds
            result_df['slack_category'] = pd.cut(
                result_df['slack_minutes'], 
                bins=[-float('inf'), self.CRITICAL_SLACK_THRESHOLD, self.URGENT_SLACK_THRESHOLD, float('inf')],
                labels=['Critical', 'Urgent', 'Safe']
            )
            
            # Identify priority containers due today
            priority_today_mask = (
                (result_df['priority'] > 0) & 
                (result_df['pull_datetime'].dt.date == current_time.date())
            )
            
            # Create a container-to-category mapping to identify unique categories per container
            container_to_category = result_df[['container_id', 'slack_category']].drop_duplicates().set_index('container_id')['slack_category']
            
            # Find containers that are priority, due today, and currently marked as 'Safe'
            priority_safe_containers = result_df.loc[priority_today_mask, 'container_id'].unique()
            # Filter to only those that are currently 'Safe'
            priority_safe_containers = [c_id for c_id in priority_safe_containers 
                                      if container_to_category.get(c_id) == 'Safe']
            
            # Update the category for these containers
            if priority_safe_containers:
                self.logger.info(f"Upgrading {len(priority_safe_containers)} priority containers from 'Safe' to 'Urgent'")
                priority_safe_mask = result_df['container_id'].isin(priority_safe_containers)
                result_df.loc[priority_safe_mask, 'slack_category'] = 'Urgent'
            
            # Add slack weights based on the slack category
            weight_mapping = {
                'Critical': self.config['slack_calculation']['weights']['critical'],
                'Urgent': self.config['slack_calculation']['weights']['urgent'],
                'Safe': self.config['slack_calculation']['weights']['safe']
            }
            result_df['slack_weight'] = result_df['slack_category'].map(weight_mapping)
            
            categorization_time = time.time() - categorization_start
            total_time = time.time() - start_time
            
            # Log performance metrics
            self.logger.debug(f"Slack calculation timing - Time calc: {time_calc_time:.3f}s, "
                             f"Lookups: {lookup_time:.3f}s, Components: {components_time:.3f}s, "
                             f"Merge: {merge_time:.3f}s, Slack calc: {slack_calc_time:.3f}s, "
                             f"Final merge: {final_merge_time:.3f}s, Categorization: {categorization_time:.3f}s, "
                             f"Total: {total_time:.3f}s")
            
            self._log_slack_summary(result_df)
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating container slack: {str(e)}")
            raise

    def _log_slack_summary(self, result_df: pd.DataFrame) -> None:
        """Log summary of slack calculations."""
        container_categories = result_df.groupby('container_id')['slack_category'].first()
        category_counts = container_categories.value_counts()
        
        summary_data = [
            [f"Critical (slack < {self.CRITICAL_SLACK_THRESHOLD} min.)",  category_counts.get('Critical', 0)],
            [f"Urgent ({self.CRITICAL_SLACK_THRESHOLD} <= slack < {self.URGENT_SLACK_THRESHOLD} min.)", category_counts.get('Urgent', 0)], 
            [f"Safe (slack >= {self.URGENT_SLACK_THRESHOLD} min.)", category_counts.get('Safe', 0)],
            ["Total Containers", len(result_df['container_id'].unique())]
        ]
        
        self.logger.info("Slack Calculation Summary:")
        self.logger.info("\n" + tabulate(summary_data, headers=['Category', 'Count'], tablefmt='grid'))  