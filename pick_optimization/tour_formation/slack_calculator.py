"""
Slack calculator module for pick optimization.

This module provides functionality to calculate slack time for containers,
i.e. the buffer time available before a container will miss its
critical pull time. Slack will be used to emphasize containers in our subproblems
"""

# Standard library imports
import logging
import math
from datetime import datetime
from typing import Dict, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import betaprime
from tabulate import tabulate

# Get module-specific logger
logger = logging.getLogger(__name__)

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
        """Create lookup tables for SKU aisles and inventory levels."""
        sku_aisle_lookup = {}
        sku_inventory_lookup = {}
        
        for sku in result_df['item_number'].unique():
            sku_data = slotbook_data[slotbook_data['item_number'] == sku]
            sku_aisles = sku_data['aisle_sequence'].unique()
            sku_aisle_lookup[sku] = sku_aisles
            
            for aisle in sku_aisles:
                inventory = sku_data[sku_data['aisle_sequence'] == aisle]['actual_qty'].sum()
                sku_inventory_lookup[(sku, aisle)] = inventory
                
        return sku_aisle_lookup, sku_inventory_lookup

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
        """Calculate the slack time for each container."""
        try:
            self.logger.info("Calculating slack time for containers...")
            result_df = container_data.copy()
            
            # Initialize time components
            result_df['time_until_pull'] = 0.0
            result_df['waiting_time'] = 0.0
            result_df['picking_time'] = 0.0
            result_df['travel_time'] = 0.0
            result_df['other_time'] = self.OTHER_TIME_BUFFER
            result_df['break_impact'] = 0.0
            result_df['slack_minutes'] = 0.0
            
            # Get configuration parameters
            buffer_wait_minutes = self.config['slack_calculation']['buffer_variability_factor'] * \
                self.config['slack_calculation']['avg_time_to_prepare_tour']
            
            # Create lookups
            sku_aisle_lookup, sku_inventory_lookup = self._create_sku_lookups(result_df, slotbook_data)
            pull_datetime_counts = container_data.groupby('pull_datetime')['container_id'].nunique().cumsum()
            
            # Process each container
            for container_id, container_items in result_df.groupby('container_id'):
                container_mask = result_df['container_id'] == container_id
                pull_datetime = container_items['pull_datetime'].iloc[0]
                
                # Calculate components
                time_until_pull = max(0, (pull_datetime - current_time).total_seconds() / 60)
                picking_time = self._calculate_picking_time(container_items)
                travel_time = self._calculate_travel_time(container_items, sku_aisle_lookup, sku_inventory_lookup)
                break_impact = self._calculate_break_impact(current_time, pull_datetime)
                
                # Calculate waiting time
                urgent_containers = pull_datetime_counts.get(pull_datetime, 0)
                queue_wait_minutes = (urgent_containers / container_target) * 60
                waiting_time = queue_wait_minutes + buffer_wait_minutes
                
                # Update result DataFrame
                result_df.loc[container_mask, 'time_until_pull'] = time_until_pull
                result_df.loc[container_mask, 'picking_time'] = picking_time
                result_df.loc[container_mask, 'travel_time'] = travel_time
                result_df.loc[container_mask, 'break_impact'] = break_impact
                result_df.loc[container_mask, 'waiting_time'] = waiting_time
                
                # Calculate slack
                total_processing_time = (
                    picking_time + travel_time + self.OTHER_TIME_BUFFER + 
                    break_impact + waiting_time
                )
                result_df.loc[container_mask, 'slack_minutes'] = time_until_pull - total_processing_time
                
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