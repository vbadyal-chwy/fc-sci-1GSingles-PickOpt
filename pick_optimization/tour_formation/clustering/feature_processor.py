"""
Feature Processing Module

This module provides the FeatureProcessor class responsible for extracting
and transforming features used in container clustering.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import pandas as pd
import numpy as np
import time


class FeatureProcessor:
    """
    Handles feature extraction and processing for container clustering.
    
    This class is responsible for transforming raw container and slotbook data
    into feature representations suitable for clustering algorithms.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the FeatureProcessor.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with feature processing parameters
        logger : logging.Logger
            Logger instance for tracking progress and errors
        """
        self.config = config
        self.logger = logger
        
        # Extract relevant configuration
        clustering_config = config.get('tour_formation', {})
        self.feature_config = clustering_config.get('feature_engineering', {})
        
        # Feature weights configuration (defaults if not specified)
        self.weights = self.feature_config.get('weights', {
            'centroid': 1.0,
            'span': 0.5,
            'distinct_aisles': 0.5
        })
        
        # Use secondary feature as span or distinct aisles
        self.use_distinct_aisles = self.feature_config.get('use_distinct_aisles', True)
        
        # Performance metrics
        self.timing_stats = {}
    
    def build_sku_aisle_mapping(self, slotbook_data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create mapping from SKUs to aisle locations.
        
        Parameters
        ----------
        slotbook_data : pd.DataFrame
            DataFrame containing slotbook information with item_number and aisle_sequence
            
        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping SKU IDs to lists of aisle numbers
        """
        start_time = time.time()
        self.logger.info("Building SKU-aisle mapping")
        
        try:
            # Verify required columns exist
            required_columns = {'item_number', 'aisle_sequence'}
            missing_columns = required_columns - set(slotbook_data.columns)
            
            if missing_columns:
                self.logger.error(f"Missing required columns in slotbook data: {missing_columns}")
                return {}
            
            # Create the mapping
            sku_aisle_mapping = {}
            
            # Group by item_number and collect unique aisle_sequence values
            for sku, group in slotbook_data.groupby('item_number'):
                aisles = sorted(group['aisle_sequence'].unique().tolist())
                sku_aisle_mapping[sku] = aisles
            
            # Log statistics
            total_skus = len(sku_aisle_mapping)
            single_location_skus = sum(1 for aisles in sku_aisle_mapping.values() if len(aisles) == 1)
            multi_location_skus = total_skus - single_location_skus
            
            if total_skus > 0:
                single_pct = (single_location_skus / total_skus) * 100
                self.logger.info(f"Total SKUs: {total_skus}")
                self.logger.info(f"Single-location SKUs: {single_location_skus} ({single_pct:.1f}%)")
                self.logger.info(f"Multi-location SKUs: {multi_location_skus} ({100-single_pct:.1f}%)")
            else:
                self.logger.warning("No SKUs found in slotbook data")
                
            self.timing_stats['build_sku_aisle_mapping'] = time.time() - start_time
            return sku_aisle_mapping
            
        except Exception as e:
            self.logger.error(f"Error building SKU-aisle mapping: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def extract_container_features(self, 
                                  container_data: pd.DataFrame,
                                  sku_aisle_mapping: Dict[str, List[int]],
                                  container_ids: Optional[List[str]] = None) -> Dict[str, Tuple[float, float, int]]:
        """
        Extract features for specified containers.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information
        sku_aisle_mapping : Dict[str, List[int]]
            Mapping of SKUs to aisle locations
        container_ids : Optional[List[str]], optional
            List of specific container IDs to process, by default None (all containers)
            
        Returns
        -------
        Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples (centroid, span, distinct_aisles)
        """
        start_time = time.time()
        
        # Select container IDs to process
        if container_ids is None:
            container_ids = container_data['container_id'].unique().tolist()
            
        self.logger.info(f"Extracting features for {len(container_ids)} containers")
        
        try:
            # Check required columns
            required_columns = {'container_id', 'item_number'}
            missing_columns = required_columns - set(container_data.columns)
            
            if missing_columns:
                self.logger.error(f"Missing required columns in container data: {missing_columns}")
                return {}
            
            # Dictionary to store features
            container_features = {}
            
            # Process each container
            for container_id in container_ids:
                
                # Get optimized aisles for this container
                container_aisles = self._get_container_aisles(container_id, container_data, sku_aisle_mapping)
                
                if not container_aisles:
                    # Skip containers with no valid aisles
                    continue
                
                # Calculate features
                centroid, span, distinct_aisles = self._compute_container_features(container_aisles)
                
                # Store features
                container_features[container_id] = (centroid, span, distinct_aisles)
            
            # Log statistics
            self.logger.info(f"Extracted features for {len(container_features)} containers")
            
            self.timing_stats['extract_container_features'] = time.time() - start_time
            return container_features
            
        except Exception as e:
            self.logger.error(f"Error extracting container features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0,1] range.
        
        Parameters
        ----------
        feature_matrix : np.ndarray
            Matrix of features with shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Normalized feature matrix
        """
        start_time = time.time()
        
        try:
            # Handle empty matrix
            if feature_matrix.size == 0 or feature_matrix.shape[0] == 0:
                self.logger.warning("Empty feature matrix provided for normalization")
                return feature_matrix
            
            # Create a copy to avoid modifying the original
            normalized = feature_matrix.copy()
            
            # Apply column-wise normalization
            for col in range(feature_matrix.shape[1]):
                col_min = np.min(feature_matrix[:, col])
                col_max = np.max(feature_matrix[:, col])
                
                # Only normalize if range is non-zero
                if col_max > col_min:
                    normalized[:, col] = (feature_matrix[:, col] - col_min) / (col_max - col_min)
                else:
                    # If all values are the same, set to 0.5
                    normalized[:, col] = 0.5
                    self.logger.debug(f"Column {col} has constant value, setting normalized values to 0.5")
            
            self.timing_stats['normalize_features'] = time.time() - start_time
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            # Return input matrix as fallback
            return feature_matrix
    
    def prepare_feature_matrix(self, 
                             container_features: Dict[str, Tuple[float, float, int]],
                             container_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert container features to feature matrix for clustering.
        
        Parameters
        ----------
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        container_ids : List[str]
            List of container IDs to include in the matrix
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Feature matrix and list of corresponding container IDs
        """
        start_time = time.time()
        self.logger.info(f"Preparing feature matrix for {len(container_ids)} containers")
        
        try:
            # Filter container IDs to those with features
            valid_container_ids = [
                c_id for c_id in container_ids 
                if c_id in container_features
            ]
            
            if not valid_container_ids:
                self.logger.warning("No valid containers with features found")
                return np.array([]), []
            
            # Prepare feature arrays
            feature_arrays = []
            final_container_ids = []
            
            for c_id in valid_container_ids:
                if c_id in container_features:
                    centroid, span, distinct_aisles = container_features[c_id]
                    
                    # Use either aisle span or distinct aisles as second feature
                    second_feature = distinct_aisles if self.use_distinct_aisles else span
                    
                    # Apply feature weights
                    weighted_centroid = centroid * self.weights.get('centroid', 1.0)
                    weighted_second = (
                        second_feature * self.weights.get('distinct_aisles' if self.use_distinct_aisles else 'span', 0.5)
                    )
                    
                    feature_arrays.append([weighted_centroid, weighted_second])
                    final_container_ids.append(c_id)
            
            # Convert to numpy array
            feature_matrix = np.array(feature_arrays)
            
            self.logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
            
            self.timing_stats['prepare_feature_matrix'] = time.time() - start_time
            return feature_matrix, final_container_ids
            
        except Exception as e:
            self.logger.error(f"Error preparing feature matrix: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([]), []
    
    def _get_container_aisles(self, 
                             container_id: str, 
                             container_data: pd.DataFrame,
                             sku_aisle_mapping: Dict[str, List[int]]) -> Set[int]:
        """
        Get optimized aisles required for a specific container.
        
        Parameters
        ----------
        container_id : str
            Container ID to process
        container_data : pd.DataFrame
            DataFrame containing container information
        sku_aisle_mapping : Dict[str, List[int]]
            Mapping of SKUs to aisles
            
        Returns
        -------
        Set[int]
            Set of aisle numbers required for this container
        """
        try:
            # Get all SKUs for this container
            container_skus = container_data[
                container_data['container_id'] == container_id
            ]['item_number'].unique()
            
            # First, identify single-location SKUs - these must be visited
            must_visit_aisles = set()
            multi_location_skus = []
            
            for sku in container_skus:
                if sku in sku_aisle_mapping:
                    aisles = sku_aisle_mapping[sku]
                    if len(aisles) == 1:
                        # Single location SKU - must visit this aisle
                        must_visit_aisles.add(aisles[0])
                    else:
                        # Multi-location SKU - will optimize later
                        multi_location_skus.append(sku)
            
            # For multi-location SKUs, choose aisles to minimize additional aisles
            for sku in multi_location_skus:
                aisles = sku_aisle_mapping[sku]
                
                # Check if any of the SKU's aisles are already in the must-visit set
                already_covered = [aisle for aisle in aisles if aisle in must_visit_aisles]
                
                if already_covered:
                    # If one or more aisles are already covered, pick the first one (no new aisles needed)
                    best_aisle = already_covered[0]
                else:
                    # Otherwise, find the aisle that minimizes the distance to the nearest must-visit aisle
                    # If no must-visit aisles yet, choose the first available aisle
                    if not must_visit_aisles:
                        best_aisle = aisles[0]
                    else:
                        # Calculate "distance" (aisle span) to the nearest must-visit aisle for each option
                        min_distance = float('inf')
                        best_aisle = None
                        
                        for aisle in aisles:
                            # Find distance to closest must-visit aisle
                            closest_distance = min(abs(aisle - existing) for existing in must_visit_aisles)
                            
                            if closest_distance < min_distance:
                                min_distance = closest_distance
                                best_aisle = aisle
                
                # Add the best aisle to the must-visit set
                must_visit_aisles.add(best_aisle)
            
            return must_visit_aisles
            
        except Exception as e:
            self.logger.error(f"Error getting container aisles for {container_id}: {str(e)}")
            return set()
    
    def _compute_container_features(self, container_aisles: Set[int]) -> Tuple[float, float, int]:
        """
        Compute feature vector for a container based on its aisles.
        
        Parameters
        ----------
        container_aisles : Set[int]
            Set of aisle numbers for this container
            
        Returns
        -------
        Tuple[float, float, int]
            Tuple containing (aisle_centroid, aisle_span, distinct_aisles)
        """
        try:
            # Convert to list and ensure it's not empty
            aisles = list(container_aisles)
            
            if not aisles:
                return 0.0, 0.0, 0
            
            # Calculate aisle centroid
            centroid = sum(aisles) / len(aisles)
            
            # Calculate aisle span
            span = max(aisles) - min(aisles) if len(aisles) > 1 else 0
            
            # Count distinct aisles
            distinct_aisles = len(aisles)
            
            return centroid, span, distinct_aisles
            
        except Exception as e:
            self.logger.error(f"Error computing container features: {str(e)}")
            return 0.0, 0.0, 0