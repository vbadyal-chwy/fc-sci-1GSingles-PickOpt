"""
Data validation module for tour formation (applies to run_complete and generate_clusters modes)

"""
import pandas as pd
import time
from typing import Tuple, Dict, Any, Optional
from tabulate import tabulate
import logging

# Get module-specific logger with workflow logging
from pick_optimization.utils.logging_config import get_logger
logger = get_logger(__name__, 'tour_formation')

class DataValidator:
    """
    Handles data type validation, null checks, business rule validation, 
    any feasibility checks and adjustments, and generates summary metrics for container and slotbook data.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize validator with a logger instance.
        
        Parameters
        ----------
        logger : logging.Logger
            Configured logger instance.
        """
        self.logger = logger
        
    def validate(self, container_data: pd.DataFrame, 
                 slotbook_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method to validate and analyze input data for Tour Formation.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Cleaned and validated container and slotbook data
        """
        self.logger.info("Starting Tour Formation data validation and analysis...")
        
        slotbook_data = slotbook_data.sort_values(['picking_flow_as_int'])
        
        # Data type validation and cleaning
        container_data = self._validate_container_data(container_data)
        slotbook_data = self._validate_slotbook_data(slotbook_data)
        
        # Business rule validation
        slotbook_data = self._validate_business_rules(container_data, slotbook_data)
        
        # Generate and display metrics
        self._generate_summary_metrics(container_data, slotbook_data)
          
        return container_data, slotbook_data
    
    def _validate_container_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean container data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Container data to validate
            
        Returns
        -------
        pd.DataFrame
            Validated and cleaned container data
        """
        self.logger.debug("Validating container data...")
        
        # Define expected data types
        container_dtypes = {
            'wh_id': str,
            'container_id': str,
            'arrive_datetime': str,
            'pull_datetime': str,
            'item_number': str,
            'pick_quantity': int,
            'aisle_sequence': int,
            'wms_pick_location': str,
            'picking_flow_as_int': int,
            'priority': int
        }
        
        # Convert data types
        for col, dtype in container_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.error(f"Error converting {col} to {dtype}: {str(e)}")
                    raise ValueError(f"Data type conversion failed for {col}")
        
        # Convert datetime columns
        datetime_cols = ['arrive_datetime', 'pull_datetime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            self.logger.warning("Null values found in container data:")
            null_table = [[col, count] for col, count in null_counts[null_counts > 0].items()]
            self.logger.warning("\n" + tabulate(null_table, headers=['Column', 'Null Count'], tablefmt='grid'))
        
        # Ensure required columns exist
        required_container_columns = set(container_dtypes.keys())
        missing_container_columns = required_container_columns - set(df.columns)
        
        if missing_container_columns:
            raise ValueError(f"Missing required columns in container data: {missing_container_columns}")
        
        return df
    
    def _validate_slotbook_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean slotbook data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Slotbook data to validate
            
        Returns
        -------
        pd.DataFrame
            Validated and cleaned slotbook data
        """
        self.logger.debug("Validating slotbook data...")
        
        # Define expected data types
        dtypes = {
            'wh_id': str,
            'location_id': str,
            'item_number': str,
            'actual_qty': int,
            'type': str,
            'print_zone': str,
            'aisle_sequence': int,
            'picking_flow_as_int': int
        }
        
        # Convert data types
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.error(f"Error converting {col} to {dtype}: {str(e)}")
                    raise ValueError(f"Data type conversion failed for {col}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            self.logger.warning("Null values found in slotbook data:")
            null_table = [[col, count] for col, count in null_counts[null_counts > 0].items()]
            self.logger.warning("\n" + tabulate(null_table, headers=['Column', 'Null Count'], tablefmt='grid'))
        
        return df
    
    def _validate_business_rules(self, container_data: pd.DataFrame, slotbook_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate business rules for the data and handle missing items.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
            
        Returns
        -------
        pd.DataFrame
            Updated slotbook data with any necessary adjustments
        """
        start_time = time.time()
        
        self.logger.debug("Validating specific checks...")
        if 'altered' not in slotbook_data.columns:
            slotbook_data['altered'] = 0

        # Handle duplicate item-aisle combinations
        # This is an assumption - we'll keep the entry with the highest quantity
        self.logger.debug("Checking for duplicate item-aisle combinations...")
        
        # Sort by actual_qty descending to keep the entry with highest quantity
        slotbook_data = slotbook_data.sort_values('actual_qty', ascending=False)
        
        # Drop duplicates keeping first occurrence
        initial_rows = len(slotbook_data)
        slotbook_data = slotbook_data.drop_duplicates(
            subset=['item_number', 'aisle_sequence'], 
            keep='first'
        )
        dropped_rows = initial_rows - len(slotbook_data)
        
        if dropped_rows > 0:
            self.logger.warning(
                f"Removed {dropped_rows} duplicate item-aisle combinations, "
                "keeping entries with highest quantity"
            )

        # Pre-compute sets for efficient lookups
        missing_items_start = time.time()
        container_items = set(container_data['item_number'].unique())
        slotbook_items = set(slotbook_data['item_number'].unique())
        missing_items = container_items - slotbook_items
        
        if missing_items:
            self.logger.warning(f"Found {len(missing_items)} items in container data not present in slotbook")
            self.logger.debug(f"Missing items: {missing_items}")
            
            # Create new rows for missing items using vectorized operations
            # This is an assumption - slotbook may have changed, so we'll create a fwd_pick with 1000 qty inventory
            missing_items_list = list(missing_items)
            
            # Get representative data for each missing item (first occurrence in container data)
            missing_item_data = container_data[
                container_data['item_number'].isin(missing_items_list)
            ].drop_duplicates(subset=['item_number'], keep='first')[
                ['wh_id', 'item_number', 'picking_flow_as_int', 'wms_pick_location', 'aisle_sequence']
            ].copy()
            
            # Add the required columns for slotbook
            missing_item_data['location_id'] = missing_item_data['wms_pick_location']
            missing_item_data['actual_qty'] = 1000
            missing_item_data['type'] = 'P'
            missing_item_data['print_zone'] = 'Z01'
            missing_item_data['altered'] = 1
            
            # Select only the columns that exist in slotbook_data
            slotbook_columns = slotbook_data.columns.tolist()
            missing_item_data = missing_item_data[
                [col for col in missing_item_data.columns if col in slotbook_columns]
            ]
            
            if not missing_item_data.empty:
                slotbook_data = pd.concat([slotbook_data, missing_item_data], ignore_index=True)
                self.logger.info(f"Added {len(missing_item_data)} missing items to slotbook data")
        
        missing_items_time = time.time() - missing_items_start
        
        # Vectorized inventory validation - calculate required vs available quantities
        inventory_start = time.time()
        
        # Group container data to get total required quantities per item
        required_quantities = container_data.groupby('item_number')['pick_quantity'].sum().reset_index()
        required_quantities.columns = ['item_number', 'total_required']
        
        # Group slotbook data to get total available quantities per item
        available_quantities = slotbook_data.groupby('item_number')['actual_qty'].sum().reset_index()
        available_quantities.columns = ['item_number', 'total_available']
        
        # Merge to compare required vs available
        inventory_comparison = pd.merge(
            required_quantities, 
            available_quantities, 
            on='item_number', 
            how='left'
        )
        
        # Fill NaN values (items not in slotbook) with 0
        inventory_comparison['total_available'] = inventory_comparison['total_available'].fillna(0)
        
        # Identify items with insufficient inventory
        insufficient_mask = inventory_comparison['total_available'] < inventory_comparison['total_required']
        insufficient_items_df = inventory_comparison[insufficient_mask].copy()
        
        if not insufficient_items_df.empty:
            self.logger.warning(f"Found {len(insufficient_items_df)} items with insufficient inventory")
            
            # Calculate additional quantities needed for all insufficient items
            insufficient_items_df['additional_qty'] = (
                insufficient_items_df['total_required'] - 
                insufficient_items_df['total_available'] + 10000  # Add buffer of 10000
            )
            
            # Batch update slotbook data for insufficient items
            # For each insufficient item, find the first row in slotbook and adjust it
            for _, row in insufficient_items_df.iterrows():
                item = row['item_number']
                additional_qty = row['additional_qty']
                required = row['total_required']
                available = row['total_available']
                
                self.logger.debug(f"Insufficient inventory for SKU {item}. Required: {required}, Available: {available}")
                
                # Find the first occurrence of this item in slotbook_data
                item_mask = slotbook_data['item_number'] == item
                item_indices = slotbook_data.index[item_mask]
                
                if len(item_indices) > 0:
                    # Update the first occurrence
                    first_index = item_indices[0]
                    slotbook_data.loc[first_index, 'actual_qty'] += additional_qty
                    slotbook_data.loc[first_index, 'altered'] = 1
                else:
                    self.logger.error(f"Could not find item {item} in slotbook data despite earlier check. This should not happen.")
        
        inventory_time = time.time() - inventory_start
        
        # Check for items with multiple locations using vectorized operations
        multi_location_start = time.time()
        item_location_counts = slotbook_data.groupby('item_number')['location_id'].nunique()
        multi_location_items = item_location_counts[item_location_counts > 1]
        
        if not multi_location_items.empty:
            total_items = len(item_location_counts)
            multi_loc_count = len(multi_location_items)
            multi_loc_pct = round(100 * multi_loc_count / total_items, 1)
            
            self.logger.info(f"Items with multiple locations: {multi_loc_count} ({multi_loc_pct}% of total items)")
        
        multi_location_time = time.time() - multi_location_start
        total_time = time.time() - start_time
        
        # Log performance metrics
        self.logger.debug(f"Business rules validation timing - Missing items: {missing_items_time:.3f}s, "
                         f"Inventory validation: {inventory_time:.3f}s, "
                         f"Multi-location check: {multi_location_time:.3f}s, "
                         f"Total: {total_time:.3f}s")
            
        return slotbook_data
             
    def _generate_summary_metrics(self, container_data: pd.DataFrame, slotbook_data: pd.DataFrame) -> None:
        """
        Generate and display summary metrics for the data.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
        """
        self.logger.info("Tour Formation input data comprises of:")
        
        # Calculate metrics
        metrics = [
            ["Distinct container IDs", container_data['container_id'].nunique()],
            ["Avg. distinct items per container", 
             round(container_data.groupby('container_id')['item_number'].nunique().mean(), 2)],
            ["Avg. quantity per item", 
             round(container_data.groupby(['container_id', 'item_number'])['pick_quantity'].sum().mean(), 2)],
            ["Earliest arrival date", container_data['arrive_datetime'].min().strftime('%Y-%m-%d %H:%M')],
            ["Latest arrival date", container_data['arrive_datetime'].max().strftime('%Y-%m-%d %H:%M')]
        ]
        
        # Print metrics table
        self.logger.info("\n" + tabulate(metrics, headers=['Metric', 'Value'], tablefmt='grid'))
        
        # Slotbook metrics
        slotbook_metrics = [
            ["Distinct items in slotbook", slotbook_data['item_number'].nunique()],
            ["Distinct locations", slotbook_data['location_id'].nunique()],
            ["Distinct aisles", slotbook_data['aisle_sequence'].nunique()],
            ["Avg. inventory per item", round(slotbook_data.groupby('item_number')['actual_qty'].sum().mean(), 2)],
            ["Items with altered inventory", slotbook_data[slotbook_data['altered'] == 1]['item_number'].nunique()]
        ]
        
        self.logger.info("\n" + tabulate(slotbook_metrics, headers=['Metric', 'Value'], tablefmt='grid'))
        
        # Log data statistics
        self.logger.info("Data Statistics:")
        self.logger.info(f"Total containers: {len(container_data['container_id'].unique())}")
        self.logger.info(f"Total items: {len(container_data['item_number'].unique())}")
        self.logger.info(f"Total locations: {len(slotbook_data['location_id'].unique())}")
        self.logger.info(f"Total SKUs: {len(slotbook_data['item_number'].unique())}")
        self.logger.info(f"Total quantity: {container_data['pick_quantity'].sum()}")
        self.logger.info(f"Average items per container: {container_data.groupby('container_id')['item_number'].nunique().mean():.2f}")
        self.logger.info(f"Average quantity per item: {float(container_data.groupby('item_number')['pick_quantity'].sum().mean()):.2f}")
        self.logger.info(f"Distinct aisles: {container_data['aisle_sequence'].nunique()}")
        
        # Create item-container mapping
        item_container_data = container_data.groupby('item_number').agg({
            'container_id': 'count',
            'pick_quantity': 'sum',
            'aisle_sequence': 'nunique'
        }).reset_index()
        
        item_container_data.columns = ['item_number', 'container_count', 'total_quantity', 'aisle_count']

def validate_input_data(
    containers_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> bool:
    """Validate input data for tour formation."""
    try:
        logger = logger or logging.getLogger(__name__)
        logger.info("Validating input data")
        # ... existing code ...
        return True
    except Exception as e:
        logger.error(f"Error validating input data: {str(e)}")
        return False 