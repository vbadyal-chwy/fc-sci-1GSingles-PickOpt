"""
Data validation module for tour formation (applies to run_complete and generate_clusters modes)

"""
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from tabulate import tabulate
import logging

# Get module-specific logger
logger = logging.getLogger(__name__)

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
            'cut_datetime': str,
            'pull_datetime': str,
            'item_number': str,
            'pick_quantity': int,
            'wms_pick_location': str,
            'print_zone': int,
            'aisle_sequence': int,
            'picking_flow_as_int': int,
            'released_flag': int,
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
        datetime_cols = ['arrive_datetime', 'cut_datetime', 'pull_datetime']
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

        # Check item numbers in container data against slotbook
        container_items = set(container_data['item_number'].unique())
        slotbook_items = set(slotbook_data['item_number'].unique())
        missing_items = container_items - slotbook_items
        
        if missing_items:
            self.logger.warning(f"Found {len(missing_items)} items in container data not present in slotbook")
            self.logger.debug(f"Missing items: {missing_items}")
            
            # Create new rows for missing items
            # This is an assumption - slotbook may have changed, so we'll create a fwd_pick with 1000 qty inventory
            new_rows = []
            for item in missing_items:
                item_container_data = container_data[container_data['item_number'] == item].iloc[0]
                
                new_row = {
                    'wh_id': item_container_data['wh_id'],
                    'item_number': item,
                    'picking_flow_as_int': item_container_data['picking_flow_as_int'],
                    'location_id': item_container_data['wms_pick_location'],
                    'aisle_sequence': item_container_data['aisle_sequence'],
                    'actual_qty': 1000,
                    'type': 'P',
                    'print_zone': 'Z01',
                    'altered': 1
                }
                new_rows.append(new_row)
            
            if new_rows:
                new_slotbook_entries = pd.DataFrame(new_rows)
                slotbook_data = pd.concat([slotbook_data, new_slotbook_entries], ignore_index=True)
                self.logger.info(f"Added {len(new_rows)} missing items to slotbook data")
        
        # Adjust inventory issues
        insufficient_items = []
        for item in container_items:
            total_required = container_data[container_data['item_number'] == item]['pick_quantity'].sum()
            total_available = slotbook_data[slotbook_data['item_number'] == item]['actual_qty'].sum()
            
            if total_available < total_required:
                insufficient_items.append((item, total_required, total_available))
                    
        if insufficient_items:
            self.logger.warning(f"Found {len(insufficient_items)} items with insufficient inventory")
                    
        for item, required, available in insufficient_items:
            self.logger.debug(f"Insufficient inventory for SKU {item}. Required: {required}, Available: {available}")

            # Locate the first row in slotbook_data where the item appears
            try:
                index_to_adjust = slotbook_data.index[slotbook_data['item_number'] == item][0]
            except IndexError:
                self.logger.error(f"Could not find item {item} in slotbook data despite earlier check. This should not happen.")
                continue # Skip this item if not found (should not occur)

            # Calculate additional inventory needed
            # This is an assumption - we don't do any real-time inventory checks/adjustments
            # Done to prevent infeasibility of the MIP model
            additional_qty = required - available + 10000  # Add buffer of 10000

            # Adjust inventory
            slotbook_data.loc[index_to_adjust, ['actual_qty', 'altered']] = [
                slotbook_data.at[index_to_adjust, 'actual_qty'] + additional_qty, 1]
            
        # Check for items with multiple locations
        item_location_counts = slotbook_data.groupby('item_number')['location_id'].nunique()
        multi_location_items = item_location_counts[item_location_counts > 1]
        
        if not multi_location_items.empty:
            total_items = len(item_location_counts)
            multi_loc_count = len(multi_location_items)
            multi_loc_pct = round(100 * multi_loc_count / total_items, 1)
            
            self.logger.info(f"Items with multiple locations: {multi_loc_count} ({multi_loc_pct}% of total items)")
            
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
            ["Latest arrival date", container_data['arrive_datetime'].max().strftime('%Y-%m-%d %H:%M')],
            ["Earliest cut date", container_data['cut_datetime'].min().strftime('%Y-%m-%d %H:%M')],
            ["Latest cut date", container_data['cut_datetime'].max().strftime('%Y-%m-%d %H:%M')]
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