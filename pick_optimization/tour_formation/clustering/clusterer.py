"""
Container Clustering Module

This module provides the main ContainerClusterer class that orchestrates
the container clustering process for the tour formation problem. It supports
writing clusters to disk for containerization.
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import math
import time
import os
from datetime import datetime

from .feature_processor import FeatureProcessor
from .clustering_engine import ClusteringEngine
from .visualization import Visualizer
from logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_formation')


class ContainerClusterer:
    """
    Main orchestration class for container clustering.
    
    This class coordinates the container clustering process, including
    critical container prioritization, feature processing, and cluster formation.
    It serves as the primary entry point for tour formation's clustering step.
    
    It supports running in containerized mode with the ability to write
    cluster definitions to output directory for distributed processing.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_dir: str, container_target: int):
        """
        Initialize the ContainerClusterer with configuration, logger, and output directory.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with clustering parameters
        logger : logging.Logger
            Logger instance for tracking progress and errors
        output_dir : str
            The base output directory for this run.
        """
        # Store configuration, logger, and output directory
        self.config = config
        self.logger = logger
        self.output_dir = output_dir # Store output directory
        
        # Extract key configuration parameters
        self.clustering_config = config.get('clustering', {})
        self.max_cluster_size = self.clustering_config['max_cluster_size']
        self.containers_per_tour = self.config['tour_formation']['max_containers_per_tour']
        self.max_picking_capacity = container_target
        
        # Create visualization path if needed
        visualization_enabled = self.clustering_config.get('generate_visualizations', False)
            
        # Initialize components, passing output_dir
        self.feature_processor = FeatureProcessor(config, logger)
        # Pass output_dir to ClusteringEngine
        self.clustering_engine = ClusteringEngine(config, logger, output_dir=self.output_dir)
        
        # Initialize Visualizer, passing a specific subdirectory if enabled
        if visualization_enabled:
            vis_output_dir = os.path.join(self.output_dir, 'cluster_visualizations') 
            self.visualizer = Visualizer(config, logger, base_output_dir=vis_output_dir)
        else:
            self.visualizer = None
        
        # Track timings for performance analysis
        self.timing_stats = {}
        
        # Track clustering results
        self.clusters = {}
        self.cluster_tours = {}
        self.cluster_metadata = {}
        
    def cluster_containers(self, container_data: pd.DataFrame, 
                           slotbook_data: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Main entry point for container clustering.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information
        slotbook_data : pd.DataFrame
            DataFrame containing slotbook/SKU information
            
        Returns
        -------
        Tuple[Dict[str, List[str]], Dict[str, int]]
            A tuple containing:
            - Dictionary mapping cluster IDs to lists of container IDs
            - Dictionary mapping cluster IDs to number of tours needed
        """
        start_time = time.time()
        self.logger.info("Starting container clustering process")
        
        try:
            # Validate input data
            if container_data.empty or slotbook_data.empty:
                self.logger.error("Empty input data provided")
                return {}, {}
                
            # Step 0: Check for critical containers
            container_ids = container_data['container_id'].unique().tolist()
            self.logger.info(f"Processing {len(container_ids)} unique containers")
            
            # Early return if all containers can fit in a single cluster
            if len(container_ids) <= self.max_cluster_size:
                self.logger.info(f"All {len(container_ids)} containers fit in a single cluster (Max Cluster Size: {self.max_cluster_size})")
                # Create single cluster with all containers
                single_cluster = {'1': container_ids}
                # Calculate tours needed
                num_tours = math.ceil(len(container_ids) / self.containers_per_tour)
                cluster_tours = {'1': num_tours}
                
                # Store results for later reference
                self.clusters = single_cluster
                self.cluster_tours = cluster_tours
                
                # Create cluster metadata
                self._create_cluster_metadata(container_data, [])
                
                # Log final statistics
                total_time = time.time() - start_time
                self.timing_stats['total_clustering_time'] = total_time
                
                self.logger.info(f"Container clustering completed in {total_time:.2f} seconds")
                self.logger.info(f"Formed 1 cluster with {len(container_ids)} containers")
                self.logger.info(f"Total tours required: {num_tours}")
                
                return single_cluster, cluster_tours
            
            critical_containers = self._identify_critical_containers(container_data)
            
            # Decision branching based on critical containers
            start_branch_time = time.time()
            
            if not critical_containers or len(container_ids) <= self.max_picking_capacity:
                # self.logger.info(
                #     f"Using standard clustering path: "
                #     f"{'No critical containers found' 
                #     if not critical_containers 
                #     else 'All containers fit within capacity'}"
                # )
                # Skip to Step 7: Form clusters from all containers
                clusters = self._handle_standard_clustering_path(
                    container_data, slotbook_data, container_ids
                )
            else:
                self.logger.info(
                    f"Using critical container prioritization path: "
                    f"Found {len(critical_containers)} critical containers"
                )
                # Follow critical container prioritization path
                clusters = self._handle_critical_container_path(
                    container_data, slotbook_data, container_ids, critical_containers
                )
                
            self.timing_stats['clustering_branch'] = time.time() - start_branch_time
            
            # Calculate tours for each cluster
            if not critical_containers or len(container_ids) <= self.max_picking_capacity:
                cluster_tours = self.clustering_engine.calculate_tours(
                    clusters, [], self.containers_per_tour
                )
            else:
                cluster_tours = self.clustering_engine.calculate_tours(
                    clusters, critical_containers, self.containers_per_tour
                )
            
            # Store results for later reference
            self.clusters = clusters
            self.cluster_tours = cluster_tours
            
            # Create cluster metadata
            self._create_cluster_metadata(container_data, critical_containers)
            
            # Log final statistics
            total_time = time.time() - start_time
            self.timing_stats['total_clustering_time'] = total_time
            
            self.logger.info(f"Container clustering completed in {total_time:.2f} seconds")
            self.logger.info(f"Formed {len(clusters)} clusters with {sum(len(c) for c in clusters.values())} containers")
            self.logger.info(f"Total tours required: {sum(cluster_tours.values())}")
            
            return clusters, cluster_tours
            
        except Exception as e:
            self.logger.error(f"Error in container clustering: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Return empty dicts in case of error
            return {}, {}
    
    def _create_cluster_metadata(self, container_data: pd.DataFrame, critical_containers: List[str]) -> None:
        """
        Create and store metadata for each cluster.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information
        critical_containers : List[str]
            List of critical container IDs
        """
        self.cluster_metadata = {}
        
        # Calculate tour offset for each cluster
        tour_offset = 0
        sorted_cluster_ids = sorted(self.clusters.keys())
        
        for cluster_id in sorted_cluster_ids:
            container_ids = self.clusters[cluster_id]
            num_tours = self.cluster_tours.get(cluster_id, 0)
            
            # Count critical containers in this cluster
            critical_count = sum(1 for c in container_ids if c in critical_containers)
            
            # Store metadata
            self.cluster_metadata[cluster_id] = {
                'cluster_id': cluster_id,
                'container_count': len(container_ids),
                'critical_container_count': critical_count,
                'num_tours': num_tours,
                'tour_id_offset': tour_offset,
                'creation_timestamp': datetime.now().isoformat()
            }
            
            # Update tour offset for the next cluster
            tour_offset += num_tours
     
            
    def _identify_critical_containers(self, container_data: pd.DataFrame) -> List[str]:
        """
        Identify critical containers based on slack category.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information with slack_category if available
            
        Returns
        -------
        List[str]
            List of critical container IDs
        """
        start_time = time.time()
        
        # Check if slack data is available
        has_slack_data = 'slack_category' in container_data.columns
        
        if not has_slack_data:
            self.logger.info("No slack_category column found in container data")
            return []
            
        # Extract container IDs with Critical or Urgent status
        try:
            # Get unique container-category pairs
            container_categories = container_data[['container_id', 'slack_category']].drop_duplicates()
            
            # Identify critical containers (Critical or Urgent)
            critical_df = container_categories[
                container_categories['slack_category'].isin(['Critical', 'Urgent'])
            ]
            critical_containers = critical_df['container_id'].unique().tolist()
            
            critical_count = len(critical_containers)
            total_count = len(container_data['container_id'].unique())
            critical_pct = 100 * critical_count / total_count if total_count > 0 else 0
            
            self.logger.info(
                f"Identified {critical_count} critical containers "
                f"({critical_pct:.1f}% of {total_count} total)"
            )
            
            # Store timing information
            self.timing_stats['identify_critical'] = time.time() - start_time
            
            return critical_containers
            
        except Exception as e:
            self.logger.error(f"Error identifying critical containers: {str(e)}")
            return []
    
    def _handle_standard_clustering_path(self, 
                                        container_data: pd.DataFrame, 
                                        slotbook_data: pd.DataFrame,
                                        container_ids: List[str]) -> Dict[str, List[str]]:
        """
        Handle clustering when no critical containers or all fit within capacity.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information
        slotbook_data : pd.DataFrame
            DataFrame containing slotbook/SKU information
        container_ids : List[str]
            List of all container IDs to process
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        """
        start_time = time.time()
        self.logger.info(f"Starting standard clustering path for {len(container_ids)} containers")
        
        try:
            # Build SKU-aisle mapping
            sku_aisle_mapping = self.feature_processor.build_sku_aisle_mapping(slotbook_data)
            
            # Extract container features
            container_features = self.feature_processor.extract_container_features(
                container_data, sku_aisle_mapping, container_ids
            )
            
            # Step 7: Form clusters from all containers
            all_container_clusters = self.clustering_engine.form_additional_clusters(
                container_ids, container_data, container_features, self.max_cluster_size
            )
            
            # Calculate how many tours we can allocate
            num_tours = math.ceil(len(container_ids)/self.containers_per_tour)
            
            # Select best clusters to fill available capacity
            selected_clusters = self.clustering_engine.select_additional_clusters(
                all_container_clusters, container_features, num_tours, self.containers_per_tour
            )
            
            # Extract just the container lists for the return value
            final_clusters = {
                cluster_id: info['containers'] 
                for cluster_id, info in selected_clusters.items()
            }
            
            # Finalize clusters (renumber and calculate statistics)
            final_clusters, cluster_stats = self.clustering_engine.finalize_clusters(
                final_clusters, 
                [],  # No critical containers in this path
                container_features,
                self.containers_per_tour
            )
            
            # Generate visualizations if enabled
            if self.visualizer:
                self.visualizer.visualize_final_clusters(
                    final_clusters, 
                    [],  # No critical containers
                    container_features
                )
            
            self.timing_stats['standard_clustering_path'] = time.time() - start_time
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in standard clustering path: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _handle_critical_container_path(self, 
                                       container_data: pd.DataFrame,
                                       slotbook_data: pd.DataFrame,
                                       container_ids: List[str],
                                       critical_containers: List[str]) -> Dict[str, List[str]]:
        """
        Handle the full critical container prioritization workflow.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            DataFrame containing container information
        slotbook_data : pd.DataFrame
            DataFrame containing slotbook/SKU information
        container_ids : List[str]
            List of all container IDs to process
        critical_containers : List[str]
            List of critical container IDs
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        """
        start_time = time.time()
        self.logger.info(
            f"Starting critical container prioritization path with "
            f"{len(critical_containers)} critical containers out of {len(container_ids)} total"
        )
        
        try:
            # Step 1: Separate containers
            non_critical_containers = [c_id for c_id in container_ids if c_id not in critical_containers]
            self.logger.info(
                f"Separated containers: {len(critical_containers)} critical, "
                f"{len(non_critical_containers)} non-critical"
            )
            
            # If critical containers exceed picking capacity, prioritize them
            if len(critical_containers) > self.max_picking_capacity:
                self.logger.warning(
                    f"Critical containers ({len(critical_containers)}) exceed "
                    f"picking capacity ({self.max_picking_capacity})"
                )
                # Just use all critical containers, no non-critical
                non_critical_containers = []
            
            # Build SKU-aisle mapping
            sku_aisle_mapping = self.feature_processor.build_sku_aisle_mapping(slotbook_data)
            
            # Extract container features for all containers
            container_features = self.feature_processor.extract_container_features(
                container_data, sku_aisle_mapping, container_ids
            )
            
            # Adjust min_clusters if needed for critical containers
            min_clusters = max(
                1,
                (len(critical_containers) // self.max_cluster_size) + 
                (1 if len(critical_containers) % self.max_cluster_size > 0 else 0)
            )
            self.logger.info(f"Adjusted minimum clusters to {min_clusters} based on critical containers")
            
            # Step 2: Form seed clusters from critical containers
            seed_clusters = self.clustering_engine.form_seed_clusters(
                critical_containers,
                container_data,
                container_features,
                self.max_cluster_size
            )
            
            if not seed_clusters:
                self.logger.error("Failed to form seed clusters from critical containers")
                return {}
            
            # Step 3: Calculate cluster centers for the seed clusters
            cluster_centers = self.clustering_engine.calculate_cluster_centers(
                seed_clusters, container_features
            )
            
            # Step 4: Run K-means with seed centers on non-critical containers
            kmeans_clusters = self.clustering_engine.kmeans_with_seed_centers(
                non_critical_containers, container_features, cluster_centers
            )
            
            # Step 5: Merge and augment clusters
            # First merge critical seed clusters with assigned non-critical containers
            merged_clusters = {}
            
            for cluster_id, center in cluster_centers.items():
                # Get critical containers in this seed cluster
                critical_in_cluster = seed_clusters[cluster_id]
                
                # Get non-critical containers assigned to this cluster
                non_critical_in_cluster = kmeans_clusters.get(cluster_id, [])
                
                # Combine them
                merged_clusters[cluster_id] = critical_in_cluster + non_critical_in_cluster
                
                self.logger.debug(
                    f"Merged cluster {cluster_id}: {len(critical_in_cluster)} critical + "
                    f"{len(non_critical_in_cluster)} non-critical"
                )
            
            # Remove non-critical containers already assigned
            assigned_non_critical = set()
            for containers in kmeans_clusters.values():
                assigned_non_critical.update(containers)
                
            remaining_non_critical = list(set(non_critical_containers) - assigned_non_critical)
            
            # Augment merged clusters with best remaining non-critical containers
            augmented_clusters, remaining_containers = self.clustering_engine.augment_clusters(
                merged_clusters,
                remaining_non_critical,
                critical_containers,
                container_features,
                self.max_cluster_size
            )
            
            # Step 6: Calculate tours for seed clusters
            cluster_tours = self.clustering_engine.calculate_tours(
                augmented_clusters, critical_containers, self.containers_per_tour
            )
            
            # Create dictionary with containers and tours
            seed_result = {
                cluster_id: {
                    'containers': containers,
                    'tours': cluster_tours[cluster_id]
                }
                for cluster_id, containers in augmented_clusters.items()
            }
            
            # Check if we've reached max_clusters with seed clusters
            total_seed_tours = sum(cluster_tours.values())
            
            if total_seed_tours >= self.max_picking_capacity // self.containers_per_tour:
                self.logger.info(
                    f"Seed clusters already use all available capacity "
                    f"({total_seed_tours} tours)"
                )
                # Extract container lists before returning
                seed_clusters_only = {
                    cluster_id: info['containers'] 
                    for cluster_id, info in seed_result.items()
                }
                
                # Finalize clusters (renumber and calculate statistics)
                final_clusters, cluster_stats = self.clustering_engine.finalize_clusters(
                    seed_clusters_only,
                    critical_containers,
                    container_features,
                    self.containers_per_tour
                )
                
                self.timing_stats['critical_path_full_capacity'] = time.time() - start_time
                return final_clusters
            
            # Step 7: Handle remaining capacity with additional clusters
            remaining_capacity = (
                self.max_picking_capacity // self.containers_per_tour
            ) - total_seed_tours
            
            final_result = seed_result
            
            if remaining_capacity > 0 and remaining_containers:
                # Form additional clusters from remaining containers
                additional_clusters = self.clustering_engine.form_additional_clusters(
                    remaining_containers, 
                    container_data, 
                    container_features, 
                    self.max_cluster_size
                )
                
                # Select best additional clusters to fill remaining capacity
                selected_additional = self.clustering_engine.select_additional_clusters(
                    additional_clusters, 
                    container_features, 
                    remaining_capacity, 
                    self.containers_per_tour
                )
                
                # Step 8: Merge seed clusters with selected additional clusters
                final_result = self.clustering_engine.merge_cluster_results(
                    seed_result, selected_additional
                )
            else:
                self.logger.info(
                    f"No remaining capacity ({remaining_capacity} tours) or "
                    f"no remaining containers ({len(remaining_containers)})"
                )
            
            # Extract just the container lists for the return value
            final_clusters = {
                cluster_id: info['containers'] 
                for cluster_id, info in final_result.items()
            }
            
            # Finalize clusters (renumber and calculate statistics)
            final_clusters, cluster_stats = self.clustering_engine.finalize_clusters(
                final_clusters,
                critical_containers,
                container_features,
                self.containers_per_tour
            )
            
            # Generate visualizations if enabled
            if self.visualizer:
                self.visualizer.visualize_final_clusters(
                    final_clusters,
                    critical_containers,
                    container_features
                )
            
            self.timing_stats['critical_container_path'] = time.time() - start_time
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in critical container path: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    