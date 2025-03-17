"""
Data preparation and preprocessing for tour formation optimization.

This module handles all data preparation tasks including:
- Container and SKU data preprocessing
- Aisle mapping and inventory tracking
- Model data structure creation
"""

from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
import pandas as pd
import logging

@dataclass
class ModelData:
    """Container for preprocessed model data"""
    container_ids: List[str]
    skus: List[str]
    container_sku_qty: Dict[Tuple[str, str], int]
    sku_aisles: Dict[str, List[int]]
    sku_min_aisle: Dict[str, List[int]]
    sku_max_aisle: Dict[str, List[int]]
    aisle_inventory: Dict[Tuple[str, int], int]
    tour_indices: List[int]  
    max_aisle: int
    is_last_iteration: bool = True
    single_location_skus: Dict[str, int] = field(default_factory=dict)  # SKU -> unique aisle
    multi_location_skus: Dict[str, List[int]] = field(default_factory=dict)  # SKU -> list of aisles
    container_fixed_aisles: Dict[str, Dict[int, int]] = field(default_factory=dict)  # container -> {aisle -> qty}

def prepare_model_data(
    container_data: pd.DataFrame,
    slotbook_data: pd.DataFrame,
    container_ids: List[str] = None,
    num_tours: int = None,
    logger: logging.Logger = None
) -> ModelData:
    """
    Prepare data structures for optimization model.
    Classify SKUs into single-location and multi-location groups.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    container_ids : Optional[List[str]]
        Optional list of specific container IDs to include in the optimization
    num_tours : Optional[int]
        Number of tours to create, if None will be calculated based on container count
    logger : Optional[logging.Logger]
        Logger instance for logging messages
        
    Returns
    -------
    ModelData
        Preprocessed data ready for model building
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        # Filter container data if specific container IDs provided
        if container_ids is not None:
            filtered_container_data = container_data[
                container_data['container_id'].isin(container_ids)
            ]
        else:
            filtered_container_data = container_data
            
        container_ids = filtered_container_data['container_id'].unique().tolist()
        skus = filtered_container_data['item_number'].unique().tolist()
        
        # Create container-SKU quantity mapping
        container_sku_qty = {}
        for _, row in filtered_container_data.iterrows():
            container_sku_qty[(row['container_id'], row['item_number'])] = row['quantity']
        
        # Create SKU-Aisle mapping and classify SKUs by location count
        sku_aisles = {}
        max_aisle = slotbook_data['aisle_sequence'].max()
        
        # Classify SKUs by number of aisle locations
        single_location_skus = {}
        multi_location_skus = {}
        
        for sku in skus:
            sku_locs = slotbook_data[
                slotbook_data['item_number'] == sku
            ]['aisle_sequence'].tolist()
            sorted_locs = sorted(sku_locs)
            sku_aisles[sku] = sorted_locs
            
            if len(sorted_locs) == 1:
                single_location_skus[sku] = sorted_locs[0]
            else:
                multi_location_skus[sku] = sorted_locs
        
        # Create mapping of fixed aisle requirements for each container
        container_fixed_aisles = {}
        for i in container_ids:
            fixed_aisles = {}
            for s in skus:
                if (i, s) in container_sku_qty and s in single_location_skus:
                    a = single_location_skus[s]
                    fixed_aisles[a] = fixed_aisles.get(a, 0) + container_sku_qty[(i, s)]
            container_fixed_aisles[i] = fixed_aisles
        
        # Log optimization statistics
        single_count = len(single_location_skus)
        multi_count = len(multi_location_skus)
        total_skus = single_count + multi_count
        single_pct = (single_count / total_skus) * 100 if total_skus > 0 else 0
        
        logger.debug(f"SKU Optimization: {single_count} SKUs ({single_pct:.1f}%) have single locations")
        logger.debug(f"SKU Optimization: {multi_count} SKUs ({100-single_pct:.1f}%) have multiple locations")
        
        # Compute SKU-specific aisle bounds
        sku_min_aisle = {}
        sku_max_aisle = {}
        for s, aisles in sku_aisles.items():
            if aisles:  # Check if the list is not empty
                sku_min_aisle[s] = min(aisles)
                sku_max_aisle[s] = max(aisles)
    
        # Create aisle inventory mapping
        aisle_inventory = {}
        for _, row in slotbook_data.iterrows():
            aisle_inventory[(row['item_number'], row['aisle_sequence'])] = row['actual_qty']
        
        # Define tour indices based on the number of tours provided or calculated
        if num_tours is None:
            # Default to number of containers if no specific number provided
            num_tours = len(container_ids)
        tour_indices = list(range(num_tours))
        
        logger.info(f"Generated {len(tour_indices)} tours for {len(container_ids)} containers")
        
        # Store preprocessed data
        return ModelData(
            container_ids=container_ids,
            skus=skus,
            container_sku_qty=container_sku_qty,
            sku_aisles=sku_aisles,
            sku_min_aisle=sku_min_aisle,
            sku_max_aisle=sku_max_aisle,
            aisle_inventory=aisle_inventory,
            tour_indices=tour_indices,
            max_aisle=max_aisle,
            single_location_skus=single_location_skus,
            multi_location_skus=multi_location_skus,
            container_fixed_aisles=container_fixed_aisles
        )
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise 