"""
Clustering Engine Module

This module provides the ClusteringEngine class responsible for
implementing various clustering algorithms and strategies for
container clustering.
"""

from typing import Dict, List, Tuple, Any
import logging
import pandas as pd
import numpy as np
import time
import math
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import os

# Import feature processor
from .feature_processor import FeatureProcessor
from logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_formation')


class ClusteringEngine:
    """
    Handles clustering operations for container clustering.
    
    This class implements various clustering algorithms and strategies,
    including hierarchical clustering, k-means, and specialized algorithms
    for critical container prioritization.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_dir: str):
        """
        Initialize the ClusteringEngine.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with clustering parameters
        logger : logging.Logger
            Logger instance for tracking progress and errors
        output_dir : str
            The base output directory for this run (e.g., output/FC/Timestamp).
        """
        self.config = config
        self.logger = logger
        self.output_dir = output_dir # Store base output dir
        
        # Extract relevant configuration
        self.clustering_params = config.get('clustering', {})
        
        # Initialize feature processor
        self.feature_processor = FeatureProcessor(config, logger)
        
        # Set up clustering parameters with defaults
        self.min_clusters = self.clustering_params.get('min_clusters', 2)
        self.max_clusters = self.clustering_params.get('max_clusters', 10)
        self.linkage_method = self.clustering_params.get('linkage_method', 'ward')
        self.max_subclustering_depth = self.clustering_params.get('subclustering', {}).get('max_depth', 3)
        
        # Initialize visualizer, passing a specific subdir within output_dir
        from .visualization import Visualizer
        vis_output_dir = os.path.join(self.output_dir, 'cluster_visualizations')
        self.visualizer = Visualizer(config, logger, base_output_dir=vis_output_dir)
        
        # Performance metrics
        self.timing_stats = {}
    
    # STEP 2: Form seed clusters from critical containers
    def form_seed_clusters(self,
                          critical_containers: List[str],
                          container_data: pd.DataFrame,
                          container_features: Dict[str, Tuple[float, float, int]],
                          max_cluster_size: int) -> Dict[int, List[str]]:
        """
        Form initial seed clusters from critical containers.
        
        Parameters
        ----------
        critical_containers : List[str]
            List of critical container IDs
        container_data : pd.DataFrame
            DataFrame containing container information
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        max_cluster_size : int
            Maximum size for any cluster
            
        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        """
        start_time = time.time()
        self.logger.info(f"Forming seed clusters from {len(critical_containers)} critical containers")
        
        try:
            # Prepare feature matrix for critical containers
            feature_arrays = []
            valid_critical_containers = []
            
            for c_id in critical_containers:
                if c_id in container_features:
                    centroid, span, distinct_aisles = container_features[c_id]
                    
                    # Use centroid and span as clustering features
                    feature_arrays.append([centroid, span])
                    valid_critical_containers.append(c_id)
            
            if len(valid_critical_containers) < 2:
                # Handle case with very few critical containers
                if len(valid_critical_containers) == 1:
                    self.logger.warning("Only one valid critical container, returning as a single cluster")
                    return {0: valid_critical_containers}
                else:
                    self.logger.error("No valid critical containers found with features")
                    return {}
            
            # Convert to numpy array
            feature_matrix = np.array(feature_arrays)
            
            # Normalize features
            normalized_features = self.feature_processor.normalize_features(feature_matrix)
            
            # Determine optimal number of clusters
            optimal_clusters = self._determine_optimal_clusters(
                normalized_features,
                min_clusters=max(2, int(len(valid_critical_containers) / max_cluster_size)),
                max_clusters=min(10, len(valid_critical_containers) // 2),
                max_cluster_size=max_cluster_size
            )
            
            self.logger.info(f"Determined optimal number of clusters: {optimal_clusters}")
            
            # Perform hierarchical clustering
            Z = linkage(normalized_features, method=self.linkage_method)
            cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
            
            # Map containers to clusters using integer IDs (0-based for consistency)
            seed_clusters: Dict[int, List[str]] = {}
            for i, container_id in enumerate(valid_critical_containers):
                # Convert 1-based fcluster label to 0-based integer ID
                cluster_id = int(cluster_labels[i]) - 1 
                
                if cluster_id not in seed_clusters:
                    seed_clusters[cluster_id] = []
                    
                seed_clusters[cluster_id].append(container_id)
            
            # Log cluster sizes
            for cluster_id, containers in seed_clusters.items():
                self.logger.info(f"Seed cluster {cluster_id}: {len(containers)} containers")
            
            # Check if any clusters exceed max_cluster_size
            large_clusters = {k: v for k, v in seed_clusters.items() if len(v) > max_cluster_size}
            
            if large_clusters:
                self.logger.info(f"Found {len(large_clusters)} seed clusters exceeding max size, \
                    applying subclustering")
                
                # Apply subclustering
                final_seed_clusters: Dict[int, List[str]] = {}
                # Start new cluster IDs from the next available integer after initial clustering
                next_cluster_id = max(seed_clusters.keys()) + 1
                
                for cluster_id, containers in seed_clusters.items():
                    if len(containers) > max_cluster_size:
                        # Apply subclustering to this cluster
                        sub_clusters, next_cluster_id = self._apply_subclustering(
                            containers,
                            container_features,
                            cluster_id, # Pass original int ID for context
                            max_cluster_size,
                            next_cluster_id, # Pass the counter
                            depth=0
                        )
                        
                        # Add subclusters to final result
                        final_seed_clusters.update(sub_clusters)
                    else:
                        # Keep smaller clusters as is
                        final_seed_clusters[cluster_id] = containers
            else:
                final_seed_clusters = seed_clusters
            
            # Generate visualization for seed clusters
            self.visualizer.visualize_seed_clusters(
                final_seed_clusters, # Pass dict with int keys
                container_features,
                critical_containers
            )
            
            self.timing_stats['form_seed_clusters'] = time.time() - start_time
            return final_seed_clusters
            
        except Exception as e:
            self.logger.error(f"Error forming seed clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            if critical_containers:
                self.logger.warning("Falling back to single seed cluster due to error")
                return {0: critical_containers} # Use integer key 0
            return {}
    
    # STEP 7B: Select additional clusters
    def select_additional_clusters(self,
                                 additional_clusters: Dict[str, List[str]],
                                 container_features: Dict[str, Tuple[float, float, int]],
                                 remaining_capacity: int,
                                 containers_per_tour: int) -> Dict[str, Dict[str, Any]]:
        """
        Select additional clusters to fill remaining capacity, prioritizing by quality.
        
        Parameters
        ----------
        additional_clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        remaining_capacity : int
            Remaining capacity in number of tours
        containers_per_tour : int
            Maximum containers per tour
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping cluster IDs to dictionaries with 'containers' and 'tours' keys
        """
        start_time = time.time()
        self.logger.info(f"Selecting additional clusters to fill remaining capacity of {remaining_capacity} tours")
        
        try:
            # Skip if no capacity or no clusters
            if remaining_capacity <= 0 or not additional_clusters:
                self.logger.info("No remaining capacity or no additional clusters to select")
                return {}
            
            # Calculate metrics for each cluster
            cluster_metrics = {}
            
            for cluster_id, containers in additional_clusters.items():
                # Calculate cluster quality
                quality = self._calculate_cluster_quality(containers, container_features)
                
                # Calculate tours needed based on total containers
                tours = math.floor(len(containers) / containers_per_tour)
                
                if tours == 0 and len(containers) > 0:
                    tours = 1  # Ensure at least one tour if cluster has containers
                
                cluster_metrics[cluster_id] = {
                    'containers': containers,
                    'quality': quality,
                    'tours': tours,
                    'size': len(containers)
                }
                
                self.logger.debug(
                    f"Cluster {cluster_id}: {len(containers)} containers, "
                    f"{tours} tours, quality={quality:.4f}"
                )
            
            # Sort by quality (higher is better)
            sorted_clusters = sorted(
                cluster_metrics.items(),
                key=lambda x: x[1]['quality'],
                reverse=True
            )
            
            # Select clusters greedily until we reach capacity
            selected = {}
            total_tours = 0
            total_containers = 0
            
            for cluster_id, info in sorted_clusters:
                if total_tours + info['tours'] <= remaining_capacity:
                    selected[cluster_id] = {
                        'containers': info['containers'],
                        'tours': info['tours']
                    }
                    total_tours += info['tours']
                    total_containers += info['size']
                    
                    self.logger.debug(
                        f"Selected cluster {cluster_id}: +{info['tours']} tours, "
                        f"total now {total_tours}/{remaining_capacity}"
                    )
                
                # Break if we've reached capacity
                if total_tours >= remaining_capacity:
                    break
            
            self.logger.info(
                f"Selected {len(selected)} additional clusters with {total_tours} tours "
                f"and {total_containers} containers"
            )
            
            # Generate visualization for selected additional clusters
            self.visualizer.visualize_additional_clusters(
                {k: v['containers'] for k, v in selected.items()},
                container_features,
                is_selected=True
            )
            
            self.timing_stats['select_additional_clusters'] = time.time() - start_time
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting additional clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _calculate_cluster_quality(self, 
                                 cluster_containers: List[str], 
                                 container_features: Dict[str, Tuple[float, float, int]]) -> float:
        """
        Calculate a quality metric for a cluster based on feature cohesion.
        
        Parameters
        ----------
        cluster_containers : List[str]
            List of container IDs in the cluster
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
            
        Returns
        -------
        float
            Quality score (higher is better)
        """
        # Extract features
        valid_containers = [c_id for c_id in cluster_containers if c_id in container_features]
        
        if len(valid_containers) < 2:
            return 0  # Quality is zero for very small clusters
        
        # Calculate centroid and dispersion
        centroids = [container_features[c_id][0] for c_id in valid_containers]
        spans = [container_features[c_id][1] for c_id in valid_containers]
        
        centroid_avg = np.mean(centroids)
        span_avg = np.mean(spans)
        
        # Calculate standard deviation (lower is better)
        centroid_std = np.std(centroids) if len(centroids) > 1 else 0
        span_std = np.std(spans) if len(spans) > 1 else 0
        
        # Calculate average distance from center (lower is better)
        centroid_dists = [abs(c - centroid_avg) for c in centroids]
        span_dists = [abs(s - span_avg) for s in spans]
        
        avg_centroid_dist = np.mean(centroid_dists)
        avg_span_dist = np.mean(span_dists)
        
        # Combine into a quality score (higher is better)
        # We want clusters with low dispersion (standard deviation)
        # and low average distance from center
        
        # Scaling factors to balance the components
        centroid_factor = 0.5
        span_factor = 0.5  # Equal weight for span
        
        # Size factor - prefer larger clusters up to max_cluster_size
        max_size = 200  # Reference size
        size_score = min(len(valid_containers) / max_size, 1.0)
        
        # Inverse of weighted sum (smaller values = higher quality)
        # Add 0.1 to avoid division by zero
        spatial_quality = 1.0 / (
            centroid_factor * (centroid_std + avg_centroid_dist) + 
            span_factor * (span_std + avg_span_dist) + 0.1
        )
        
        # Combine spatial quality with size factor
        quality = spatial_quality * (0.5 + 0.5 * size_score)
        
        return quality
    
    # STEP 8: Merge cluster results
    def merge_cluster_results(self, 
                            seed_clusters: Dict[str, Dict[str, Any]], 
                            additional_clusters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Merge seed clusters and additional clusters into final result.
        
        Parameters
        ----------
        seed_clusters : Dict[str, Dict[str, Any]]
            Dictionary of seed clusters with containers and tours
        additional_clusters : Dict[str, Dict[str, Any]]
            Dictionary of additional clusters with containers and tours
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Combined dictionary of all clusters
        """
        start_time = time.time()
        self.logger.info(
            f"Merging {len(seed_clusters)} seed clusters with "
            f"{len(additional_clusters)} additional clusters"
        )
        
        try:
            # Start with seed clusters, assigning sequential IDs from 1
            final_clusters = {}
            for i, (_, info) in enumerate(seed_clusters.items(), 1):
                final_clusters[str(i)] = info
            
            # Add additional clusters with continuing sequential IDs
            next_id = len(seed_clusters) + 1
            for info in additional_clusters.values():
                final_clusters[str(next_id)] = info
                next_id += 1
            
            # Count total containers and tours
            total_containers = sum(len(info['containers']) for info in final_clusters.values())
            total_tours = sum(info['tours'] for info in final_clusters.values())
            
            self.logger.info(
                f"Final result: {len(final_clusters)} clusters with "
                f"{total_containers} containers and {total_tours} tours"
            )
            
            # Extract container lists for visualization
            '''final_cluster_containers = {
                cluster_id: info['containers'] 
                for cluster_id, info in final_clusters.items()
            }'''
            
            # Get critical containers from seed clusters
            critical_containers = []
            for info in seed_clusters.values():
                critical_containers.extend(info['containers'])
            
            '''# Generate visualization for final merged clusters
            self.visualizer.visualize_final_clusters(
                final_cluster_containers,
                critical_containers,
                self.feature_processor.container_features
            )'''
            
            self.timing_stats['merge_cluster_results'] = time.time() - start_time
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error merging cluster results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return seed_clusters
    
    # Finalize clusters
    def finalize_clusters(self, 
                        clusters: Dict[str, List[str]], 
                        critical_containers: List[str],
                        container_features: Dict[str, Tuple[float, float, int]],
                        containers_per_tour: int) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Finalize clusters with sequential IDs and statistics.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Original clusters with arbitrary IDs
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        containers_per_tour : int
            Maximum containers per tour
            
        Returns
        -------
        Tuple[Dict[str, List[str]], pd.DataFrame]
            Renumbered clusters with sequential IDs and DataFrame with cluster statistics
        """
        start_time = time.time()
        self.logger.info("Finalizing clusters with sequential IDs and statistics")
        
        try:
            # Convert critical containers to a set for efficient lookups
            critical_set = set(critical_containers)
            
            # Sort clusters by their minimum aisle (to keep a logical order)
            sorted_clusters = []
            
            for cluster_id, containers in clusters.items():
                # Calculate minimum aisle for sorting
                min_aisle = float('inf')
                for c_id in containers:
                    if c_id in container_features:
                        centroid, _, _ = container_features[c_id]
                        min_aisle = min(min_aisle, centroid)
                
                # Use a default high value if no valid containers found
                if min_aisle == float('inf'):
                    min_aisle = 9999
                    
                sorted_clusters.append((cluster_id, containers, min_aisle))
            
            # Sort by minimum aisle
            sorted_clusters.sort(key=lambda x: x[2])
            
            # Renumber the clusters sequentially
            renumbered_clusters = {}
            cluster_stats = []
            
            for idx, (old_id, containers, _) in enumerate(sorted_clusters, 1):
                new_id = str(idx)
                renumbered_clusters[new_id] = containers
                
                # Count critical and non-critical containers
                critical_count = sum(1 for c_id in containers if c_id in critical_set)
                non_critical_count = len(containers) - critical_count
                
                # Calculate tours based on the specified logic
                if critical_count > 0:
                    # For clusters with critical containers, round up based on critical containers
                    tours = max(1, (critical_count + containers_per_tour - 1) // containers_per_tour)
                else:
                    # For clusters without critical containers, simply divide total count
                    tours = max(1, len(containers) // containers_per_tour)
                
                # Calculate cluster metrics
                centroids = [container_features[c_id][0] for c_id in containers if c_id in container_features]
                spans = [container_features[c_id][1] for c_id in containers if c_id in container_features]
                distinct_aisles = [container_features[c_id][2] for c_id in containers if c_id in container_features]
                
                if centroids:
                    avg_centroid = sum(centroids) / len(centroids)
                    min_centroid = min(centroids)
                    max_centroid = max(centroids)
                else:
                    avg_centroid = min_centroid = max_centroid = 0
                    
                if spans:
                    avg_span = sum(spans) / len(spans)
                else:
                    avg_span = 0
                    
                if distinct_aisles:
                    avg_distinct_aisles = sum(distinct_aisles) / len(distinct_aisles)
                else:
                    avg_distinct_aisles = 0
                
                # Store stats for this cluster
                cluster_stats.append({
                    'ClusterID': new_id,
                    'TotalContainers': len(containers),
                    'CriticalContainers': critical_count,
                    'NonCriticalContainers': non_critical_count,
                    'CriticalPercentage': 100 * critical_count / len(containers) if containers else 0,
                    'AvgCentroid': avg_centroid,
                    'MinCentroid': min_centroid,
                    'MaxCentroid': max_centroid,
                    'AvgSpan': avg_span,
                    'AvgDistinctAisles': avg_distinct_aisles,
                    'NumTours': tours
                })
            
            # Convert cluster_stats to a DataFrame
            cluster_stats_df = pd.DataFrame(cluster_stats)
            
            # Display statistics in tabular format
            try:
                from tabulate import tabulate
                
                # Prepare data for tabulation
                headers = ["Cluster ID", "Total", "Critical", "Non-Critical", "Critical %", "Tours", "Min Aisle", "Max Aisle", "Avg Centroid", "Avg Span"]
                table_data = []
                
                for stats in cluster_stats:
                    table_data.append([
                        stats['ClusterID'],
                        stats['TotalContainers'],
                        stats['CriticalContainers'],
                        stats['NonCriticalContainers'],
                        f"{stats['CriticalPercentage']:.1f}%",
                        stats['NumTours'],
                        f"{stats['MinCentroid']:.1f}",
                        f"{stats['MaxCentroid']:.1f}",
                        f"{stats['AvgCentroid']:.1f}",
                        f"{stats['AvgSpan']:.1f}"
                    ])
                
                # Calculate totals
                total_containers = sum(len(containers) for containers in renumbered_clusters.values())
                total_critical = sum(sum(1 for c_id in containers if c_id in critical_set) 
                                   for containers in renumbered_clusters.values())
                total_non_critical = total_containers - total_critical
                critical_pct = 100 * total_critical / total_containers if total_containers else 0
                total_tours = sum(stats['NumTours'] for stats in cluster_stats)
                
                # Add a totals row
                table_data.append([
                    "TOTAL",
                    total_containers,
                    total_critical,
                    total_non_critical,
                    f"{critical_pct:.1f}%",
                    total_tours,
                    "-",
                    "-",
                    "-",
                    "-"
                ])
                
                # Generate the table with grid format for better readability
                table = tabulate(table_data, headers=headers, tablefmt="grid")
                self.logger.info("\nDetailed Cluster Statistics Table:")
                self.logger.info("\n" + table)
                
            except ImportError:
                # Fallback to simpler display if tabulate isn't available
                self.logger.info("\nCluster Statistics Summary:")
                for stats in cluster_stats:
                    self.logger.info(f"Cluster {stats['ClusterID']}: {stats['TotalContainers']} containers "
                                   f"({stats['CriticalContainers']} critical, {stats['NonCriticalContainers']} non-critical) - {stats['NumTours']} tours")
            
            # Overall statistics
            self.logger.info("\nOverall Statistics:")
            self.logger.info(f"Total clusters: {len(renumbered_clusters)}")
            self.logger.info(f"Total containers: {total_containers}")
            self.logger.info(f"Critical containers: {total_critical} ({100 * total_critical / total_containers:.1f}%)")
            self.logger.info(
                f"Non-critical containers: {total_containers - total_critical} "
                f"({100 * (total_containers - total_critical) / total_containers:.1f}%)"
            )
            self.logger.info(f"Total tours required: {total_tours}")
            
            # Generate visualization for final clusters
            self.visualizer.visualize_final_clusters(
                renumbered_clusters,
                critical_containers,
                container_features
            )
            
            self.timing_stats['finalize_clusters'] = time.time() - start_time
            return renumbered_clusters, cluster_stats_df
            
        except Exception as e:
            self.logger.error(f"Error finalizing clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return clusters, pd.DataFrame()
    
    # Helper method to determine optimal number of clusters
    def _determine_optimal_clusters(self, 
                                  feature_matrix: np.ndarray,
                                  min_clusters: int,
                                  max_clusters: int,
                                  max_cluster_size: int) -> int:
        """
        Determine optimal number of clusters using silhouette analysis.
        
        Parameters
        ----------
        feature_matrix : np.ndarray
            Feature matrix for clustering
        min_clusters : int
            Minimum number of clusters to consider
        max_clusters : int
            Maximum number of clusters to consider
        max_cluster_size : int
            Maximum size for any cluster
            
        Returns
        -------
        int
            Optimal number of clusters
        """
        try:
            # Adjust max_clusters based on data size
            max_clusters = min(max_clusters, feature_matrix.shape[0] // 2)
            max_clusters = max(min_clusters + 1, max_clusters)  # Ensure at least 2 values to compare
            
            # Special case for small datasets - just use min_clusters
            if feature_matrix.shape[0] < min_clusters * 3:
                return min_clusters
                
            self.logger.debug(f"Performing silhouette analysis for {min_clusters} to {max_clusters} clusters")
            
            # Calculate silhouette scores for different numbers of clusters
            silhouette_scores = {}
            
            for n_clusters in range(min_clusters, max_clusters + 1):
                # Perform hierarchical clustering
                Z = linkage(feature_matrix, method=self.linkage_method)
                cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
                
                # Ensure we have at least 2 clusters (silhouette score requires at least 2)
                unique_clusters = np.unique(cluster_labels)
                if len(unique_clusters) < 2:
                    self.logger.warning(f"Only found {len(unique_clusters)} unique clusters for n_clusters={n_clusters}, skipping")
                    continue
                
                # Calculate silhouette score
                try:
                    score = silhouette_score(feature_matrix, cluster_labels)
                    silhouette_scores[n_clusters] = score
                    
                    self.logger.debug(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
                except Exception as e:
                    self.logger.warning(f"Error calculating silhouette score for {n_clusters} clusters: {str(e)}")
            
            # Find optimal number of clusters
            if not silhouette_scores:
                self.logger.warning("No valid silhouette scores found, defaulting to min_clusters")
                return min_clusters
            
            optimal_clusters = max(silhouette_scores.items(), key=lambda x: x[1])[0]
            self.logger.info(f"Optimal number of clusters: {optimal_clusters} with score {silhouette_scores[optimal_clusters]:.4f}")
            
            return optimal_clusters
            
        except Exception as e:
            self.logger.error(f"Error determining optimal clusters: {str(e)}")
            return min_clusters
    
    # STEP 3: Calculate cluster centers
    def calculate_cluster_centers(self,
                                clusters: Dict[str, List[str]],
                                container_features: Dict[str, Tuple[float, float, int]]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate centers for each cluster.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
            
        Returns
        -------
        Dict[str, Tuple[float, float]]
            Dictionary mapping cluster IDs to centers (centroid, span)
        """
        start_time = time.time()
        self.logger.info("Calculating cluster centers")
        
        try:
            centers = {}
            
            for cluster_id, container_ids in clusters.items():
                valid_containers = [c_id for c_id in container_ids if c_id in container_features]
                
                if not valid_containers:
                    self.logger.warning(f"Cluster {cluster_id} has no valid containers with features")
                    # Provide default center
                    centers[cluster_id] = (0.0, 0.0)
                    continue
                    
                # Calculate average centroid and span
                centroids = [container_features[c_id][0] for c_id in valid_containers]
                spans = [container_features[c_id][1] for c_id in valid_containers]
                
                avg_centroid = np.mean(centroids)
                avg_span = np.mean(spans)
                
                centers[cluster_id] = (avg_centroid, avg_span)
                
                self.logger.debug(
                    f"Cluster {cluster_id} center: centroid={avg_centroid:.2f}, span={avg_span:.2f}"
                )
            
            self.logger.info(f"Calculated centers for {len(centers)} clusters")
            
            self.timing_stats['calculate_cluster_centers'] = time.time() - start_time
            return centers
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster centers: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    # STEP 4: K-means with seed centers
    def kmeans_with_seed_centers(self,
                               non_critical_containers: List[str],
                               container_features: Dict[str, Tuple[float, float, int]],
                               cluster_centers: Dict[str, Tuple[float, float]]) -> Dict[str, List[str]]:
        """
        Run K-means with seed centers on non-critical containers.
        
        Parameters
        ----------
        non_critical_containers : List[str]
            List of non-critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        cluster_centers : Dict[str, Tuple[float, float]]
            Dictionary mapping cluster IDs to centers
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        """
        start_time = time.time()
        self.logger.info(f"Running fixed-center clustering on {len(non_critical_containers)} non-critical containers")
        
        try:
            # Extract features for non-critical containers
            feature_arrays = []
            valid_containers = []
            
            for c_id in non_critical_containers:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    feature_arrays.append([centroid, span])
                    valid_containers.append(c_id)
            
            if not valid_containers:
                self.logger.warning("No valid non-critical containers with features found")
                return {k: [] for k in cluster_centers.keys()}
            
            # Convert list of features to numpy array
            feature_matrix = np.array(feature_arrays)
            
            # Map cluster IDs to indices and build fixed centers array
            cluster_id_to_idx = {cid: i for i, cid in enumerate(cluster_centers.keys())}
            idx_to_cluster_id = {i: cid for cid, i in cluster_id_to_idx.items()}
            
            k = len(cluster_centers)
            fixed_centers = np.zeros((k, 2))
            for cluster_id, (centroid, span) in cluster_centers.items():
                idx = cluster_id_to_idx[cluster_id]
                fixed_centers[idx] = [centroid, span]
            
            # Single pass assignment: assign each container to the closest fixed seed center
            assignments = np.zeros(len(valid_containers), dtype=int)
            
            for i in range(len(valid_containers)):
                distances = np.linalg.norm(feature_matrix[i] - fixed_centers, axis=1)
                closest_cluster = np.argmin(distances)
                assignments[i] = closest_cluster
            
            # Build final clusters mapping cluster ID to assigned container IDs
            final_clusters = {cid: [] for cid in cluster_centers.keys()}
            for i, container_id in enumerate(valid_containers):
                cluster_idx = assignments[i]
                cluster_id = idx_to_cluster_id[cluster_idx]
                final_clusters[cluster_id].append(container_id)
            
            # Report cluster sizes
            for cluster_id, containers in final_clusters.items():
                self.logger.info(f"K-means cluster {cluster_id}: {len(containers)} containers")
            
            self.timing_stats['kmeans_with_seed_centers'] = time.time() - start_time
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in K-means with seed centers: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {k: [] for k in cluster_centers.keys()}
    
    # STEP 5: Augment clusters
    def augment_clusters(self,
                        seed_clusters: Dict[str, List[str]],
                        non_critical_containers: List[str],
                        critical_containers: List[str],
                        container_features: Dict[str, Tuple[float, float, int]],
                        max_cluster_size: int) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Augment seed clusters with non-critical containers.
        
        Parameters
        ----------
        seed_clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        non_critical_containers : List[str]
            List of non-critical container IDs available for assignment
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        max_cluster_size : int
            Maximum size for any cluster
            
        Returns
        -------
        Tuple[Dict[str, List[str]], List[str]]
            Dictionary mapping cluster IDs to lists of container IDs and
            list of remaining unassigned containers
        """
        start_time = time.time()
        self.logger.info(f"Augmenting {len(seed_clusters)} clusters with non-critical containers")
        
        try:
            # Convert critical containers to a set for faster lookups
            critical_set = set(critical_containers)
            
            # Calculate cluster centers for feature proximity calculations
            cluster_centers = self.calculate_cluster_centers(seed_clusters, container_features)
            
            # Prepare final clusters
            final_clusters = {k: v.copy() for k, v in seed_clusters.items()}
            
            # Track remaining non-critical containers
            remaining = set(non_critical_containers)
            
            # For each cluster, identify critical containers and calculate space left
            for cluster_id, containers in final_clusters.items():
                # Count critical containers in this cluster
                critical_in_cluster = [c_id for c_id in containers if c_id in critical_set]
                non_critical_in_cluster = [c_id for c_id in containers if c_id not in critical_set]
                
                critical_count = len(critical_in_cluster)
                current_total = len(containers)
                
                # Space left is based on critical containers only, not total
                space_left = max_cluster_size - critical_count
                
                self.logger.info(
                    f"Cluster {cluster_id}: critical={critical_count}, non-critical={len(non_critical_in_cluster)}, " 
                    f"total={current_total}, space left={space_left}"
                )
                
                # If no space left based on critical containers, remove all non-critical
                if space_left <= 0:
                    self.logger.info(f"Cluster {cluster_id} already exceeds max size with critical containers")
                    # Remove all non-critical containers from this cluster
                    final_clusters[cluster_id] = critical_in_cluster
                    # Add removed containers back to remaining
                    for c_id in non_critical_in_cluster:
                        remaining.add(c_id)
                    continue
                
                # If we already have non-critical containers, check if we need to remove some
                if len(non_critical_in_cluster) > space_left:
                    self.logger.info(
                        f"Cluster {cluster_id} has too many non-critical containers, "
                        f"removing {len(non_critical_in_cluster) - space_left}"
                    )
                    
                    # Calculate distances to cluster center for existing non-critical containers
                    center = cluster_centers[cluster_id]
                    nc_distances = []
                    
                    for c_id in non_critical_in_cluster:
                        if c_id in container_features:
                            centroid, span, _ = container_features[c_id]
                            distance = np.sqrt((centroid - center[0])**2 + (span - center[1])**2)
                            nc_distances.append((c_id, distance))
                    
                    # Sort by distance (closest first)
                    nc_distances.sort(key=lambda x: x[1])
                    
                    # Keep only the closest non-critical containers
                    keep_containers = [c_id for c_id, _ in nc_distances[:space_left]]
                    
                    # Remove the extra non-critical containers
                    remove_containers = [c_id for c_id in non_critical_in_cluster if c_id not in keep_containers]
                    
                    # Update the cluster
                    final_clusters[cluster_id] = critical_in_cluster + keep_containers
                    
                    # Add removed containers back to remaining
                    for c_id in remove_containers:
                        remaining.add(c_id)
                        
                    # Update space left
                    space_left = max_cluster_size - len(final_clusters[cluster_id])
                else:
                    # We still have space for more non-critical containers
                    space_left = space_left - len(non_critical_in_cluster)
                
                # If we still have space, add more non-critical containers
                if space_left > 0:
                    self.logger.info(f"Cluster {cluster_id} can add {space_left} more non-critical containers")
                    
                    # Calculate distance from each remaining container to this center
                    center = cluster_centers[cluster_id]
                    container_distances = []
                    
                    for c_id in remaining:
                        if c_id in container_features:
                            centroid, span, _ = container_features[c_id]
                            distance = np.sqrt((centroid - center[0])**2 + (span - center[1])**2)
                            container_distances.append((c_id, distance))
                    
                    # Sort by distance (closest first)
                    container_distances.sort(key=lambda x: x[1])
                    
                    # Add closest containers
                    containers_to_add = [c_id for c_id, _ in container_distances[:space_left]]
                    final_clusters[cluster_id].extend(containers_to_add)
                    
                    self.logger.info(f"Added {len(containers_to_add)} additional non-critical containers to cluster {cluster_id}")
                    
                    # Remove from remaining pool
                    for c_id in containers_to_add:
                        if c_id in remaining:
                            remaining.remove(c_id)
            
            # Log final cluster sizes
            for cluster_id, containers in final_clusters.items():
                critical_count = sum(1 for c_id in containers if c_id in critical_set)
                non_critical_count = len(containers) - critical_count
                
                self.logger.info(
                    f"Augmented cluster {cluster_id}: {critical_count} critical + "
                    f"{non_critical_count} non-critical = {len(containers)} total"
                )
            
            remaining_list = list(remaining)
            self.logger.info(f"Augmentation completed: {len(remaining_list)} non-critical containers remain unassigned")
            
            # Generate visualization for augmented clusters
            self.visualizer.visualize_augmented_clusters(
                final_clusters,
                critical_containers,
                container_features
            )
            
            # Generate comparison visualization
            self.visualizer.visualize_cluster_comparison(
                seed_clusters,
                final_clusters,
                critical_containers,
                container_features
            )
            
            self.timing_stats['augment_clusters'] = time.time() - start_time
            return final_clusters, remaining_list
            
        except Exception as e:
            self.logger.error(f"Error augmenting clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return seed_clusters, non_critical_containers
    
    # STEP 6: Calculate tours
    def calculate_tours(self,
                       clusters: Dict[int, List[str]],
                       critical_containers: List[str],
                       containers_per_tour: int) -> Dict[int, int]:
        """
        Calculate tours for each cluster based on critical containers.
        
        Parameters
        ----------
        clusters : Dict[int, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        critical_containers : List[str]
            List of critical container IDs
        containers_per_tour : int
            Maximum containers per tour
            
        Returns
        -------
        Dict[int, int]
            Dictionary mapping cluster IDs to number of tours
        """
        start_time = time.time()
        self.logger.debug("Calculating tours for clusters")
        
        try:
            critical_set = set(critical_containers)
            cluster_tours = {}
            total_tours = 0
            
            for cluster_id, container_ids in clusters.items():
                # Count critical containers in this cluster
                critical_count = sum(1 for c_id in container_ids if c_id in critical_set)
                
                if critical_count > 0:
                    # For clusters with critical containers, round up based on critical containers
                    tours = max(1, (critical_count + containers_per_tour - 1) // containers_per_tour)
                else:
                    # For clusters without critical containers, divide and round down
                    tours = max(1, math.floor(len(container_ids) / containers_per_tour))
                
                # Store number of tours
                cluster_tours[cluster_id] = tours
                total_tours += tours
                
                self.logger.debug(f"Cluster {cluster_id}: {critical_count} critical containers, {tours} tours")
            
            self.logger.info(f"Total tours across all clusters: {total_tours}")
            
            self.timing_stats['calculate_tours'] = time.time() - start_time
            return cluster_tours
            
        except Exception as e:
            self.logger.error(f"Error calculating tours: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Fallback: assign 1 tour per cluster
            return {cluster_id: 1 for cluster_id in clusters.keys()}
    
    # STEP 7A: Form additional clusters
    def form_additional_clusters(self,
                               remaining_containers: List[str],
                               container_data: pd.DataFrame,
                               container_features: Dict[str, Tuple[float, float, int]],
                               max_cluster_size: int) -> Dict[str, List[str]]:
        """
        Form additional clusters from remaining containers.
        
        Parameters
        ----------
        remaining_containers : List[str]
            List of remaining container IDs
        container_data : pd.DataFrame
            DataFrame containing container information
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        max_cluster_size : int
            Maximum size for any cluster
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        """
        start_time = time.time()
        self.logger.info(f"Forming additional clusters from {len(remaining_containers)} remaining containers")
        
        try:
            # Skip if no remaining containers
            if not remaining_containers:
                self.logger.info("No remaining containers to cluster")
                return {}
                
            # Get feature matrix for remaining containers
            feature_arrays = []
            valid_containers = []
            
            for c_id in remaining_containers:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    feature_arrays.append([centroid, span])
                    valid_containers.append(c_id)
            
            if not valid_containers:
                self.logger.warning("No valid remaining containers with features")
                return {}
            
            # Convert to numpy array
            feature_matrix = np.array(feature_arrays)
            
            # Normalize features
            normalized_features = self.feature_processor.normalize_features(feature_matrix)
            
            # Estimate optimal number of clusters based on max_cluster_size
            estimated_clusters = max(2, len(valid_containers) // max_cluster_size + 1)
            max_possible_clusters = min(10, len(valid_containers) // 2)
            
            # Determine optimal number of clusters
            optimal_clusters = self._determine_optimal_clusters(
                normalized_features,
                min_clusters=2,
                max_clusters=min(max_possible_clusters, estimated_clusters),
                max_cluster_size=max_cluster_size
            )
            
            self.logger.info(f"Determined optimal number of additional clusters: {optimal_clusters}")
            
            # Perform hierarchical clustering
            Z = linkage(normalized_features, method=self.linkage_method)
            cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
            
            # Map containers to clusters
            additional_clusters = {}
            for i, container_id in enumerate(valid_containers):
                cluster_id = int(cluster_labels[i])
                cluster_key = f"additional_{cluster_id}"
                
                if cluster_key not in additional_clusters:
                    additional_clusters[cluster_key] = []
                    
                additional_clusters[cluster_key].append(container_id)
            
            # Iteratively break down clusters that exceed max_cluster_size
            final_clusters = {}
            clusters_to_process = additional_clusters
            iteration = 0
            max_iterations = 10  # Safety limit to prevent infinite loops
            
            while clusters_to_process and iteration < max_iterations:
                iteration += 1
                self.logger.info(f"Iteration {iteration}: Processing {len(clusters_to_process)} clusters")
                
                next_clusters_to_process = {}
                
                for cluster_id, containers in clusters_to_process.items():
                    # If the cluster is small enough, add it to the final results
                    if len(containers) <= max_cluster_size:
                        final_clusters[cluster_id] = containers
                        self.logger.debug(f"Cluster {cluster_id} size {len(containers)} is within limit")
                        continue
                        
                    # Otherwise, subdivide this cluster
                    self.logger.info(f"Subdividing cluster {cluster_id} with {len(containers)} containers")
                    
                    # Initialize cluster ID counter for first call or use the running counter
                    current_cluster_id_counter = getattr(self, '_current_cluster_id_counter', 1)
                    
                    # Apply subclustering
                    sub_clusters, updated_counter = self._apply_subclustering(
                        containers,
                        container_features,
                        cluster_id,
                        max_cluster_size,
                        current_cluster_id_counter,
                        depth=0
                    )
                    
                    # Update the counter for next iterations
                    self._current_cluster_id_counter = updated_counter
                    
                    # Process subclusters
                    for sub_id, sub_containers in sub_clusters.items():
                        # If subcluster is small enough, add to final results
                        if len(sub_containers) <= max_cluster_size:
                            final_clusters[sub_id] = sub_containers
                            self.logger.debug(f"Subcluster {sub_id} size {len(sub_containers)} is within limit")
                        else:
                            # Otherwise, add to next iteration for further subdivision
                            next_clusters_to_process[sub_id] = sub_containers
                            self.logger.debug(f"Subcluster {sub_id} size {len(sub_containers)} needs further subdivision")
                
                clusters_to_process = next_clusters_to_process
                
                # Check if we have clusters that still need processing
                if not next_clusters_to_process:
                    self.logger.info(f"All clusters are within size limit after iteration {iteration}")
                    break
            
            # Handle any remaining large clusters if we hit the iteration limit
            if clusters_to_process:
                self.logger.warning(
                    f"Reached max iterations ({max_iterations}), "
                    f"{len(clusters_to_process)} clusters still exceed size limit"
                )
                # Add remaining clusters to the final result regardless of size
                for cluster_id, containers in clusters_to_process.items():
                    final_clusters[f"{cluster_id}_forced"] = containers
            
            # Log final statistics
            cluster_sizes = [len(containers) for containers in final_clusters.values()]
            if cluster_sizes:
                self.logger.info(f"Final additional clusters: {len(final_clusters)}")
                self.logger.info(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
                self.logger.info(f"Max cluster size: {max(cluster_sizes)}")
                self.logger.info(f"Min cluster size: {min(cluster_sizes)}")
                
                # Check if all clusters are within limit
                oversized = [size for size in cluster_sizes if size > max_cluster_size]
                if oversized:
                    self.logger.warning(f"{len(oversized)} clusters still exceed max_cluster_size")
                else:
                    self.logger.info("All clusters are within max_cluster_size limit")

            # Relabel clusters with sequential IDs
            sequential_clusters = {}
            for idx, (_, containers) in enumerate(sorted(final_clusters.items(), key=lambda x: len(x[1]), reverse=True), 1):
                sequential_clusters[str(idx)] = containers
            final_clusters = sequential_clusters
            
            # Generate visualization for additional clusters (before selection)
            self.visualizer.visualize_additional_clusters(
                final_clusters,
                container_features,
                is_selected=False
            )
            
            self.timing_stats['form_additional_clusters'] = time.time() - start_time
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error forming additional clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    # Helper method for subclustering
    def _apply_subclustering(self,
                           containers: List[str],
                           container_features: Dict[str, Tuple[float, float, int]],
                           parent_cluster_id: int,
                           max_cluster_size: int,
                           current_cluster_id_counter: int,
                           depth: int = 0) -> Tuple[Dict[int, List[str]], int]:
        """
        Recursively apply subclustering to a cluster until all subclusters
        are within the max_cluster_size limit.

        Parameters
        ----------
        containers : List[str]
            List of container IDs in the cluster to subcluster
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        parent_cluster_id : int
            The ID of the parent cluster being subclustered (for context)
        max_cluster_size : int
            Maximum size for any subcluster
        current_cluster_id_counter : int
            The next available integer ID to assign to a new subcluster.
        depth : int, optional
            Current recursion depth, by default 0

        Returns
        -------
        Tuple[Dict[int, List[str]], int]
            A dictionary mapping new integer cluster IDs to container lists,
            and the updated cluster ID counter.
        """
        if depth >= self.max_subclustering_depth:
            self.logger.warning(
                f"Max subclustering depth ({self.max_subclustering_depth}) reached for parent {parent_cluster_id}. "
                f"Returning oversized cluster with {len(containers)} containers."
            )
            # Assign the next available ID to this oversized cluster
            new_cluster_id = current_cluster_id_counter
            return {new_cluster_id: containers}, current_cluster_id_counter + 1

        # Prepare features for subclustering
        sub_feature_arrays = []
        valid_sub_containers = []
        for c_id in containers:
            if c_id in container_features:
                centroid, span, _ = container_features[c_id]
                sub_feature_arrays.append([centroid, span])
                valid_sub_containers.append(c_id)

        if len(valid_sub_containers) <= 1:
            # Cannot subcluster further, assign the next ID
            new_cluster_id = current_cluster_id_counter
            return {new_cluster_id: valid_sub_containers}, current_cluster_id_counter + 1

        sub_feature_matrix = np.array(sub_feature_arrays)
        normalized_sub_features = self.feature_processor.normalize_features(sub_feature_matrix)

        # Determine number of clusters for subclustering (start with 2)
        num_sub_clusters = 2
        
        # Perform clustering (start with k=2)
        Z_sub = linkage(normalized_sub_features, method=self.linkage_method)
        sub_cluster_labels = fcluster(Z_sub, num_sub_clusters, criterion='maxclust') # 1-based labels

        # Map containers to subclusters
        sub_clusters_temp = {}
        for i, container_id in enumerate(valid_sub_containers):
            sub_label = int(sub_cluster_labels[i]) # Temporary 1-based label
            if sub_label not in sub_clusters_temp:
                sub_clusters_temp[sub_label] = []
            sub_clusters_temp[sub_label].append(container_id)

        # Recursively apply subclustering if needed and assign final IDs
        final_sub_clusters: Dict[int, List[str]] = {}
        updated_cluster_id_counter = current_cluster_id_counter

        for temp_label, sub_containers in sub_clusters_temp.items():
            if len(sub_containers) > max_cluster_size:
                self.logger.debug(f"Recursively subclustering part of parent {parent_cluster_id} (depth {depth+1}) with {len(sub_containers)} containers")
                # Recursively call subclustering
                recursive_clusters, updated_cluster_id_counter = self._apply_subclustering(
                    sub_containers,
                    container_features,
                    parent_cluster_id, # Pass parent ID for context
                    max_cluster_size,
                    updated_cluster_id_counter, # Pass the current counter
                    depth + 1
                )
                final_sub_clusters.update(recursive_clusters)
            elif sub_containers: # Only add non-empty clusters
                # Assign the next available integer ID
                new_cluster_id = updated_cluster_id_counter
                final_sub_clusters[new_cluster_id] = sub_containers
                updated_cluster_id_counter += 1
            else:
                # Handle case of empty subcluster (should not happen with valid containers)
                self.logger.warning(f"Empty subcluster generated for parent {parent_cluster_id} at depth {depth}")
                
        return final_sub_clusters, updated_cluster_id_counter