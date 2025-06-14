"""
Data transformation module for Pick Optimization.

This module handles transforming external data from Snowflake queries
to match the SQLite database schema format.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict


class DataTransformer:
    """Transforms external data to match SQLite schema."""
    
    def __init__(self, execution_id: str):
        """
        Initialize with execution ID for data isolation.
        
        Parameters
        ----------
        execution_id : str
            Unique execution ID for this simulation run
        """
        self.execution_id = execution_id
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"DataTransformer initialized for execution: {execution_id}")
    
    def transform_containers(self, raw_data: pd.DataFrame, wh_id: str) -> pd.DataFrame:
        """
        Transform container master data to match containers table schema.
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw container data from Snowflake
        wh_id : str
            Warehouse ID
            
        Returns
        -------
        pd.DataFrame
            Transformed data matching containers table schema
        """
        self.logger.info(f"Transforming container master data: {len(raw_data)} rows")
        self.logger.debug(f"Using execution_id: {self.execution_id} (type: {type(self.execution_id)})")
        
        if raw_data.empty:
            self.logger.warning("No container master data to transform")
            return pd.DataFrame()
        
        # Create transformed dataframe with required columns
        transformed = pd.DataFrame(index=raw_data.index)
        
        # Required fields - set execution_id first
        transformed['execution_id'] = self.execution_id
        transformed['container_id'] = raw_data.get('CONTAINER_ID', '')
        transformed['wh_id'] = wh_id
        
        # Optional fields with default values
        transformed['priority'] = raw_data.get('PRIORITY', 1)
        transformed['che_route'] = raw_data.get('CHE_ROUTE', '')
        transformed['arrive_datetime'] = pd.to_datetime(raw_data.get('ARRIVE_DATETIME'), errors='coerce')
        transformed['original_promised_pull_datetime'] = pd.to_datetime(
            raw_data.get('ORIGINAL_PROMISED_PULL_DATETIME'), errors='coerce'
        )
        
        # Default values for operational fields
        transformed['released_flag'] = False
        transformed['tour_id'] = None
        transformed['release_datetime'] = None
        transformed['created_at'] = datetime.now()
        transformed['updated_at'] = datetime.now()
        
        # Debug: Check execution_id before filtering
        self.logger.debug(f"Execution_id column before filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        # Remove rows with missing container_id
        transformed = transformed[transformed['container_id'].notna() & (transformed['container_id'] != '')]
        
        # Debug: Check execution_id after filtering
        self.logger.debug(f"Execution_id column after filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        self.logger.info(f"Container transformation completed: {len(transformed)} rows")
        return transformed
    
    def transform_container_details(self, raw_data: pd.DataFrame, wh_id: str) -> pd.DataFrame:
        """
        Transform container details to match container_details table schema.
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw container details data from Snowflake
        wh_id : str
            Warehouse ID
            
        Returns
        -------
        pd.DataFrame
            Transformed data matching container_details table schema
        """
        self.logger.info(f"Transforming container details data: {len(raw_data)} rows")
        self.logger.debug(f"Using execution_id: {self.execution_id} (type: {type(self.execution_id)})")
        
        if raw_data.empty:
            self.logger.warning("No container details data to transform")
            return pd.DataFrame()
        
        # Create transformed dataframe with proper index
        transformed = pd.DataFrame(index=raw_data.index)
        
        # Required fields - set execution_id first
        transformed['execution_id'] = self.execution_id
        transformed['container_id'] = raw_data.get('CONTAINER_ID', '')
        transformed['pick_id'] = raw_data.get('PICK_ID', '')
        transformed['item_number'] = raw_data.get('ITEM_NUMBER', '')
        transformed['wh_id'] = wh_id
        
        # Optional fields
        transformed['planned_quantity'] = pd.to_numeric(raw_data.get('PLANNED_QUANTITY'), errors='coerce')
        transformed['pick_location'] = raw_data.get('PICK_LOCATION', '')
        transformed['location_status'] = raw_data.get('LOCATION_STATUS', '')
        transformed['aisle_sequence'] = pd.to_numeric(raw_data.get('AISLE_SEQUENCE'), errors='coerce')
        transformed['aisle_name'] = raw_data.get('AISLE_NAME', '')
        transformed['picking_flow_as_int'] = pd.to_numeric(raw_data.get('PICKING_FLOW_AS_INT'), errors='coerce')
        transformed['created_at'] = datetime.now()
        
        # Debug: Check execution_id before filtering
        self.logger.debug(f"Execution_id column before filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        # Remove rows with missing required fields
        required_fields = ['container_id', 'pick_id', 'item_number']
        for field in required_fields:
            transformed = transformed[transformed[field].notna() & (transformed[field] != '')]
        
        # Debug: Check execution_id after filtering
        self.logger.debug(f"Execution_id column after filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        self.logger.info(f"Container details transformation completed: {len(transformed)} rows")
        return transformed
    
    def transform_slotbook(self, raw_data: pd.DataFrame, wh_id: str) -> pd.DataFrame:
        """
        Transform slotbook data to match slotbook table schema.
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw slotbook data from Snowflake
        wh_id : str
            Warehouse ID
            
        Returns
        -------
        pd.DataFrame
            Transformed data matching slotbook table schema
        """
        self.logger.info(f"Transforming slotbook data: {len(raw_data)} rows")
        self.logger.debug(f"Using execution_id: {self.execution_id} (type: {type(self.execution_id)})")
        
        if raw_data.empty:
            self.logger.warning("No slotbook data to transform")
            return pd.DataFrame()
        
        # Create transformed dataframe with proper index
        transformed = pd.DataFrame(index=raw_data.index)
        
        # Required fields - set execution_id first
        transformed['execution_id'] = self.execution_id
        transformed['inventory_snapshot_date'] = pd.to_datetime(
            raw_data.get('INVENTORY_SNAPSHOT_DATE'), errors='coerce'
        ).dt.date
        transformed['item_number'] = raw_data.get('ITEM_NUMBER', '')
        transformed['location_id'] = raw_data.get('LOCATION_ID', '')
        transformed['wh_id'] = wh_id
        
        # Optional fields
        transformed['aisle_sequence'] = pd.to_numeric(raw_data.get('AISLE_SEQUENCE'), errors='coerce')
        transformed['picking_flow_as_int'] = pd.to_numeric(raw_data.get('PICKING_FLOW_AS_INT'), errors='coerce')
        transformed['actual_qty'] = pd.to_numeric(raw_data.get('ACTUAL_QTY'), errors='coerce')
        transformed['type'] = raw_data.get('TYPE', '')
        transformed['created_at'] = datetime.now()
        
        # Debug: Check execution_id before filtering
        self.logger.debug(f"Execution_id column before filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        # Remove rows with missing required fields
        required_fields = ['item_number', 'location_id']
        for field in required_fields:
            transformed = transformed[transformed[field].notna() & (transformed[field] != '')]
        
        # Remove rows with null inventory_snapshot_date
        transformed = transformed[transformed['inventory_snapshot_date'].notna()]
        
        # Debug: Check execution_id after filtering
        self.logger.debug(f"Execution_id column after filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        self.logger.info(f"Slotbook transformation completed: {len(transformed)} rows")
        return transformed
    
    def transform_labor_data(self, raw_data: pd.DataFrame, wh_id: str) -> pd.DataFrame:
        """
        Transform labor data to match labor_data table schema.
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw labor data from Vertica
        wh_id : str
            Warehouse ID
            
        Returns
        -------
        pd.DataFrame
            Transformed data matching labor_data table schema
        """
        self.logger.info(f"Transforming labor data: {len(raw_data)} rows")
        self.logger.debug(f"Using execution_id: {self.execution_id} (type: {type(self.execution_id)})")
        
        if raw_data.empty:
            self.logger.warning("No labor data to transform")
            return pd.DataFrame()
        
        # Create transformed dataframe with proper index
        transformed = pd.DataFrame(index=raw_data.index)
        
        # Required fields - set execution_id first
        transformed['execution_id'] = self.execution_id
        transformed['wh_id'] = wh_id
        transformed['date'] = pd.to_datetime(raw_data.get('date'), errors='coerce').dt.date
        transformed['hour'] = pd.to_numeric(raw_data.get('hour'), errors='coerce')
        transformed['minutes'] = pd.to_numeric(raw_data.get('minutes'), errors='coerce')
        transformed['count_employees'] = pd.to_numeric(raw_data.get('count_employees'), errors='coerce')
        transformed['created_at'] = datetime.now()
        
        # Debug: Check execution_id before filtering
        self.logger.debug(f"Execution_id column before filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        # Remove rows with missing required fields
        transformed = transformed[transformed['date'].notna()]
        transformed = transformed[transformed['hour'].notna()]
        transformed = transformed[transformed['minutes'].notna()]
        
        # Debug: Check execution_id after filtering
        self.logger.debug(f"Execution_id column after filtering: {transformed['execution_id'].iloc[0] if len(transformed) > 0 else 'NO DATA'}")
        
        self.logger.info(f"Labor data transformation completed: {len(transformed)} rows")
        return transformed
    
    def transform_all_data(self, raw_data: Dict[str, pd.DataFrame], wh_id: str) -> Dict[str, pd.DataFrame]:
        """
        Transform all extracted data to match SQLite schema.
        
        Parameters
        ----------
        raw_data : Dict[str, pd.DataFrame]
            Dictionary containing raw data from external sources:
            - 'container_master': Container master data (Snowflake)
            - 'container_details': Container details data (Snowflake)
            - 'slotbook': Slotbook data (Snowflake)
            - 'labor_data': Labor data (Vertica)
        wh_id : str
            Warehouse ID
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing transformed data ready for SQLite import
        """
        self.logger.info("Starting data transformation for all datasets")
        
        transformed_data = {}
        
        # Transform container master data
        if 'container_master' in raw_data:
            transformed_data['containers'] = self.transform_containers(
                raw_data['container_master'], wh_id
            )
        
        # Transform container details
        if 'container_details' in raw_data:
            transformed_data['container_details'] = self.transform_container_details(
                raw_data['container_details'], wh_id
            )
        
        # Transform slotbook data
        if 'slotbook' in raw_data:
            transformed_data['slotbook'] = self.transform_slotbook(
                raw_data['slotbook'], wh_id
            )
        
        # Transform labor data
        if 'labor_data' in raw_data:
            transformed_data['labor_data'] = self.transform_labor_data(
                raw_data['labor_data'], wh_id
            )
        
        # Log transformation summary
        for key, df in transformed_data.items():
            self.logger.info(f"Transformed {key}: {len(df)} rows")
        
        self.logger.info("Data transformation completed for all datasets")
        return transformed_data
    
    def validate_transformed_data(self, transformed_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate transformed data for basic consistency.
        
        Parameters
        ----------
        transformed_data : Dict[str, pd.DataFrame]
            Transformed data to validate
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        self.logger.info("Validating transformed data")
        
        validation_passed = True
        
        # Check if we have containers data
        if 'containers' not in transformed_data or transformed_data['containers'].empty:
            self.logger.error("No container data found - this is required")
            validation_passed = False
        
        # Check container details reference valid containers
        if 'container_details' in transformed_data and 'containers' in transformed_data:
            container_ids = set(transformed_data['containers']['container_id'])
            detail_container_ids = set(transformed_data['container_details']['container_id'])
            
            orphaned_details = detail_container_ids - container_ids
            if orphaned_details:
                self.logger.warning(f"Found {len(orphaned_details)} container details with no matching container")
                # Remove orphaned details
                transformed_data['container_details'] = transformed_data['container_details'][
                    transformed_data['container_details']['container_id'].isin(container_ids)
                ]
        
        # Check for duplicate container IDs
        if 'containers' in transformed_data:
            containers_df = transformed_data['containers']
            duplicate_containers = containers_df[containers_df.duplicated(['execution_id', 'container_id'])]
            if not duplicate_containers.empty:
                self.logger.error(f"Found {len(duplicate_containers)} duplicate container IDs")
                validation_passed = False
        
        if validation_passed:
            self.logger.info("Data validation passed")
        else:
            self.logger.error("Data validation failed")
        
        return validation_passed 