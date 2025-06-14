"""
Visualization Module

This module provides the Visualizer class responsible for
generating visualizations for different stages of the
container clustering process.
"""

from typing import Dict, List, Tuple,  Any
import logging
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from pick_optimization.utils.logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_formation')
matplotlib.use('Agg')


class Visualizer:
    """
    Handles visualization for container clustering.
    
    This class generates various visualizations to help understand
    and analyze the clustering process and its results.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, base_output_dir: str):
        """
        Initialize the Visualizer.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with visualization parameters
        logger : logging.Logger
            Logger instance for tracking progress and errors
        base_output_dir : str
            The base directory where visualization subdirectories will be created.
        """
        self.config = config
        self.logger = logger
        self.base_output_dir = base_output_dir # Store the provided base directory
        self.dpi = 300
        self.clustering_config = config.get('clustering', {})
        self.enabled = self.clustering_config.get('generate_visualizations', False)

        # --- Create output subdirectories relative to base_output_dir --- 
        self.seed_clusters_path = os.path.join(self.base_output_dir, 'seed_clusters')
        self.augmented_clusters_path = os.path.join(self.base_output_dir, 'augmented_clusters')
        self.additional_clusters_path = os.path.join(self.base_output_dir, 'additional_clusters')
        self.final_clusters_path = os.path.join(self.base_output_dir, 'final_clusters')
        self.stats_path = os.path.join(self.base_output_dir, 'stats')
        
        # Create all directories
        for path in [self.seed_clusters_path, self.augmented_clusters_path, 
                     self.additional_clusters_path, self.final_clusters_path,
                     self.stats_path]:
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                self.logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
                self.enabled = False # Disable visualization if dir creation fails
                return # Stop initialization if essential dirs can't be made
        
        # Set matplotlib style for consistent visualizations
        plt.style.use('ggplot')
        
        # Performance tracking
        self.timing_stats = {}
    
    def visualize_seed_clusters(self, 
                              clusters: Dict[str, List[str]],
                              container_features: Dict[str, Tuple[float, float, int]],
                              critical_containers: List[str],
                              output_path: str = None
                              ) -> None:
        """
        Visualize initial seed clusters.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        critical_containers : List[str]
            List of critical container IDs
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        if not self.enabled:
            return
            
        start_time = time.time()
        
        if not clusters:
            self.logger.warning("No clusters to visualize for seed clusters")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                output_path = os.path.join(self.seed_clusters_path, 'seed_clusters.png')
                
            self.logger.debug(f"Generating seed clusters visualization: {output_path}")
            
            # Create a set of critical containers for faster lookups
            critical_set = set(critical_containers)
            
            # Create the figure
            plt.figure(figsize=(16, 10))
            
            # Create a colormap with enough colors for all clusters
            num_clusters = len(clusters)
            cmap = plt.cm.get_cmap('tab20', num_clusters)
            
            # Create a mapping from cluster IDs to indices for coloring
            cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(clusters.keys())}
            
            # Prepare data per cluster for legend and scatter plot
            cluster_data = {}
            
            for cluster_id, container_ids in clusters.items():
                cluster_idx = cluster_id_to_idx[cluster_id]
                color = cmap(cluster_idx)
                
                # Initialize data structures for this cluster
                cluster_data[cluster_id] = {
                    'centroids': [],
                    'spans': [],
                    'is_critical': [],
                    'color': color
                }
                
                # Process containers
                for container_id in container_ids:
                    if container_id in container_features:
                        centroid, span, _ = container_features[container_id]
                        is_critical = container_id in critical_set
                        
                        cluster_data[cluster_id]['centroids'].append(centroid)
                        cluster_data[cluster_id]['spans'].append(span)
                        cluster_data[cluster_id]['is_critical'].append(is_critical)
            
            # Create custom legend handles
            legend_handles = []
            
            # Plot each cluster separately
            for cluster_id, data in cluster_data.items():
                # Skip clusters with no valid data
                if not data['centroids']:
                    continue
                    
                # Plot non-critical containers with 'o' marker
                non_critical_mask = [not x for x in data['is_critical']]
                if any(non_critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if non_critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if non_critical_mask[i]],
                        color=data['color'],
                        marker='o',
                        s=50,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5
                    )
                
                # Plot critical containers with 'X' marker
                critical_mask = data['is_critical']
                if any(critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if critical_mask[i]],
                        color=data['color'],
                        marker='X',
                        s=100,
                        alpha=0.9,
                        edgecolors='black',
                        linewidths=1.0
                    )
                
                # Create legend handle for this cluster
                legend_handles.append(
                    mpatches.Patch(color=data['color'], label=f'Cluster {cluster_id}')
                )
                
                # Calculate cluster centroid for annotation
                avg_centroid = sum(data['centroids']) / len(data['centroids'])
                avg_span = sum(data['spans']) / len(data['spans'])
                
                # Count critical containers in this cluster
                critical_count = sum(data['is_critical'])
                total_count = len(data['centroids'])
                
                # Annotate cluster ID and stats at cluster centroid
                plt.annotate(
                    f"{cluster_id}: {critical_count}/{total_count}",
                    xy=(avg_centroid, avg_span),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=10,
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add markers to legend for critical and non-critical
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=10, label='Non-Critical')
            )
            legend_handles.append(
                Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', 
                       markersize=10, label='Critical')
            )
            
            # Add labels and title
            plt.xlabel('Aisle Centroid', fontsize=12)
            plt.ylabel('Aisle Span', fontsize=12)
            plt.title('Final Cluster Assignments', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add legend in two parts - clusters and markers
            if num_clusters <= 20:  # Only show legend for a reasonable number of clusters
                plt.legend(
                    handles=legend_handles,
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=min(5, num_clusters + 2),  # Distribute in columns
                    frameon=True,
                    fontsize=10
                )
            
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Generate statistics visualization (Update its default path too)
            stats_output_path = os.path.join(self.stats_path, 'seed_cluster_stats.png')
            self.visualize_cluster_stats(
                clusters, 
                critical_containers,
                container_features,
                stats_output_path
            )
            
            self.timing_stats['visualize_seed_clusters'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing final clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def visualize_cluster_stats(self, 
                              clusters: Dict[str, List[str]],
                              critical_containers: List[str],
                              container_features: Dict[str, Tuple[float, float, int]],
                              output_path: str = None) -> None:
        """
        Generate statistics visualization for clusters.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        start_time = time.time()
        
        if not self.enabled:
            return
            
        if not clusters:
            self.logger.warning("No clusters to visualize for cluster statistics")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                # Default path for general stats if not specified
                output_path = os.path.join(self.stats_path, 'general_cluster_stats.png')
                
            self.logger.debug(f"Generating cluster statistics visualization: {output_path}")
            
            # Create a set of critical containers for faster lookups
            critical_set = set(critical_containers)
            
            # Calculate statistics per cluster
            cluster_stats = []
            
            for cluster_id, container_ids in clusters.items():
                valid_containers = [c_id for c_id in container_ids if c_id in container_features]
                if not valid_containers:
                    continue
                    
                total_containers = len(valid_containers)
                critical_count = sum(1 for c_id in valid_containers if c_id in critical_set)
                non_critical_count = total_containers - critical_count
                critical_pct = 100 * critical_count / total_containers if total_containers else 0
                
                # Calculate average centroid and span
                centroids = [container_features[c_id][0] for c_id in valid_containers]
                spans = [container_features[c_id][1] for c_id in valid_containers]
                
                avg_centroid = sum(centroids) / len(centroids) if centroids else 0
                avg_span = sum(spans) / len(spans) if spans else 0
                
                # Add min/max aisle values
                min_aisle = min(centroids) if centroids else 0
                max_aisle = max(centroids) if centroids else 0
                aisle_range = max_aisle - min_aisle if centroids else 0
                
                cluster_stats.append({
                    'ClusterID': cluster_id,
                    'TotalContainers': total_containers,
                    'CriticalContainers': critical_count,
                    'NonCriticalContainers': non_critical_count,
                    'CriticalPercentage': critical_pct,
                    'AvgCentroid': avg_centroid,
                    'MinAisle': min_aisle,
                    'MaxAisle': max_aisle,
                    'AisleRange': aisle_range,
                    'AvgSpan': avg_span
                })
            
            # Sort clusters by ID
            cluster_stats.sort(key=lambda x: x['ClusterID'])
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
            
            # Bar colors
            #num_clusters = len(cluster_stats)
            #colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
            
            # Plot 1: Container counts (stacked bars)
            x = range(len(cluster_stats))
            cluster_ids = [stats['ClusterID'] for stats in cluster_stats]
            critical_counts = [stats['CriticalContainers'] for stats in cluster_stats]
            non_critical_counts = [stats['NonCriticalContainers'] for stats in cluster_stats]
            
            # Plot stacked bars
            ax1.bar(x, critical_counts, label='Critical', color='red', alpha=0.7)
            ax1.bar(x, non_critical_counts, bottom=critical_counts, label='Non-Critical', color='blue', alpha=0.7)
            
            # Add labels
            for i, stats in enumerate(cluster_stats):
                # Total label at top
                ax1.text(i, stats['TotalContainers'] + 1, str(stats['TotalContainers']), 
                       ha='center', va='bottom', fontsize=9)
                
                # Critical label in middle of its section
                if stats['CriticalContainers'] > 0:
                    ax1.text(i, stats['CriticalContainers'] / 2, str(stats['CriticalContainers']), 
                           ha='center', va='center', color='white', fontsize=9, fontweight='bold')
                
                # Non-critical label in middle of its section
                if stats['NonCriticalContainers'] > 0:
                    ax1.text(i, stats['CriticalContainers'] + stats['NonCriticalContainers'] / 2, str(stats['NonCriticalContainers']), 
                           ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            
            ax1.set_title('Container Counts by Cluster', fontsize=14)
            ax1.set_xlabel('Cluster ID', fontsize=12)
            ax1.set_ylabel('Number of Containers', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(cluster_ids)
            ax1.legend(loc='upper right')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Critical percentage and average metrics
            critical_pcts = [stats['CriticalPercentage'] for stats in cluster_stats]
            
            # Create a second axis for the centroid and span
            ax3 = ax2.twinx()
            
            # Plot critical percentage as bars
            # bars = ax2.bar(x, critical_pcts, color=colors, alpha=0.7)
            
            # Plot average centroid and span as lines
            avg_centroids = [stats['AvgCentroid'] for stats in cluster_stats]
            avg_spans = [stats['AvgSpan'] for stats in cluster_stats]
            
            line1, = ax3.plot(x, avg_centroids, 'o-', color='green', linewidth=2, label='Avg Centroid')
            line2, = ax3.plot(x, avg_spans, 's-', color='purple', linewidth=2, label='Avg Span')
            
            # Add percentage labels on bars
            for i, pct in enumerate(critical_pcts):
                ax2.text(i, pct + 2, f"{pct:.1f}%", ha='center', va='bottom', fontsize=9)
            
            ax2.set_title('Critical Container Percentage and Average Metrics', fontsize=14)
            ax2.set_xlabel('Cluster ID', fontsize=12)
            ax2.set_ylabel('Critical Container Percentage (%)', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(cluster_ids)
            ax2.set_ylim(0, 110)  # Leave room for labels
            
            ax3.set_ylabel('Average Value', fontsize=12)
            
            # Create a combined legend for both line plots
            lines = [line1, line2]
            labels = [line.get_label() for line in lines]
            ax3.legend(lines, labels, loc='upper right')
            
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            # Create additional bar chart for aisle ranges
            self._visualize_aisle_ranges(cluster_stats, os.path.join(os.path.dirname(output_path), 'aisle_ranges.png'))
            
            self.timing_stats['visualize_cluster_stats'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing cluster statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _visualize_aisle_ranges(self, 
                              cluster_stats: List[Dict[str, Any]],
                              output_path: str) -> None:
        """
        Generate visualization of aisle ranges per cluster.
        
        Parameters
        ----------
        cluster_stats : List[Dict[str, Any]]
            List of cluster statistics dictionaries
        output_path : str
            Path to save visualization
        """
        try:
            plt.figure(figsize=(14, 8))
            
            # Extract data
            cluster_ids = [stats['ClusterID'] for stats in cluster_stats]
            min_aisles = [stats['MinAisle'] for stats in cluster_stats]
            max_aisles = [stats['MaxAisle'] for stats in cluster_stats]
            
            # Generate x positions
            x = np.arange(len(cluster_ids))
            width = 0.35
            
            # Plot bars
            plt.bar(x - width/2, min_aisles, width, label='Min Aisle', color='skyblue')
            plt.bar(x + width/2, max_aisles, width, label='Max Aisle', color='salmon')
            
            # Add labels and grid
            plt.xlabel('Cluster ID', fontsize=12)
            plt.ylabel('Aisle Number', fontsize=12)
            plt.title('Aisle Ranges by Cluster', fontsize=14)
            plt.xticks(x, cluster_ids)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add aisle range annotations
            for i, stats in enumerate(cluster_stats):
                plt.annotate(
                    f"Range: {stats['AisleRange']:.1f}",
                    xy=(i, max_aisles[i] + 0.5),
                    ha='center',
                    fontsize=9
                )
            
            # Save figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error visualizing aisle ranges: {str(e)}")
    
    def visualize_augmented_clusters(self, 
                                   clusters: Dict[str, List[str]],
                                   critical_containers: List[str],
                                   container_features: Dict[str, Tuple[float, float, int]],
                                   output_path: str = None) -> None:
        """
        Visualize clusters after augmentation.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        start_time = time.time()
        
        if not self.enabled:
            return
            
        if not clusters:
            self.logger.warning("No clusters to visualize for augmented clusters")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                output_path = os.path.join(self.augmented_clusters_path, 'augmented_clusters.png')
                
            self.logger.debug(f"Generating augmented clusters visualization: {output_path}")
            
            # Create a set of critical containers for faster lookups
            critical_set = set(critical_containers)
            
            # Create a figure
            plt.figure(figsize=(16, 10))
            
            # Create a colormap with enough colors for all clusters
            num_clusters = len(clusters)
            cmap = plt.cm.get_cmap('tab20', num_clusters)
            
            # Create a mapping from cluster IDs to indices for coloring
            cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(clusters.keys())}
            
            # Prepare data per cluster for legend and scatter plot
            cluster_data = {}
            
            for cluster_id, container_ids in clusters.items():
                cluster_idx = cluster_id_to_idx[cluster_id]
                color = cmap(cluster_idx)
                
                # Initialize data structures for this cluster
                cluster_data[cluster_id] = {
                    'centroids': [],
                    'spans': [],
                    'is_critical': [],
                    'color': color
                }
                
                # Process containers
                for container_id in container_ids:
                    if container_id in container_features:
                        centroid, span, _ = container_features[container_id]
                        is_critical = container_id in critical_set
                        
                        cluster_data[cluster_id]['centroids'].append(centroid)
                        cluster_data[cluster_id]['spans'].append(span)
                        cluster_data[cluster_id]['is_critical'].append(is_critical)
            
            # Create custom legend handles
            legend_handles = []
            
            # Plot each cluster separately
            for cluster_id, data in cluster_data.items():
                # Skip clusters with no valid data
                if not data['centroids']:
                    continue
                    
                # Plot non-critical containers with 'o' marker
                non_critical_mask = [not x for x in data['is_critical']]
                if any(non_critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if non_critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if non_critical_mask[i]],
                        color=data['color'],
                        marker='o',
                        s=50,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5
                    )
                
                # Plot critical containers with 'X' marker
                critical_mask = data['is_critical']
                if any(critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if critical_mask[i]],
                        color=data['color'],
                        marker='X',
                        s=100,
                        alpha=0.9,
                        edgecolors='black',
                        linewidths=1.0
                    )
                
                # Create legend handle for this cluster
                legend_handles.append(
                    mpatches.Patch(color=data['color'], label=f'Cluster {cluster_id}')
                )
                
                # Calculate cluster centroid for annotation
                avg_centroid = sum(data['centroids']) / len(data['centroids'])
                avg_span = sum(data['spans']) / len(data['spans'])
                
                # Count critical containers in this cluster
                critical_count = sum(data['is_critical'])
                total_count = len(data['centroids'])
                
                # Annotate cluster ID and stats at cluster centroid
                plt.annotate(
                    f"{cluster_id}: {critical_count}/{total_count}",
                    xy=(avg_centroid, avg_span),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=10,
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add markers to legend for critical and non-critical
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=10, label='Non-Critical')
            )
            legend_handles.append(
                Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', 
                       markersize=10, label='Critical')
            )
            
            # Add labels and title
            plt.xlabel('Aisle Centroid', fontsize=12)
            plt.ylabel('Aisle Span', fontsize=12)
            plt.title('Clusters After Augmentation', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add legend in two parts - clusters and markers
            if num_clusters <= 20:  # Only show legend for a reasonable number of clusters
                plt.legend(
                    handles=legend_handles,
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=min(5, num_clusters + 2),  # Distribute in columns
                    frameon=True,
                    fontsize=10
                )
            
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Generate statistics visualization
            stats_output_path = os.path.join(self.stats_path, 'augmented_cluster_stats.png')
            self.visualize_cluster_stats(
                clusters, 
                critical_containers,
                container_features,
                stats_output_path
            )
            
            self.timing_stats['visualize_augmented_clusters'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing augmented clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def visualize_cluster_comparison(self, 
                                   before_clusters: Dict[str, List[str]],
                                   after_clusters: Dict[str, List[str]],
                                   critical_containers: List[str],
                                   container_features: Dict[str, Tuple[float, float, int]],
                                   output_path: str = None) -> None:
        """
        Create comparison visualization between cluster states.
        
        Parameters
        ----------
        before_clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs (before)
        after_clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs (after)
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        start_time = time.time()
        
        if not self.enabled:
            return
            
        if not before_clusters or not after_clusters:
            self.logger.warning("Missing cluster data for comparison visualization")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                output_path = os.path.join(self.augmented_clusters_path, 'cluster_comparison.png')
                
            self.logger.debug(f"Generating cluster comparison visualization: {output_path}")
            
            # Create a set of critical containers for faster lookups
            critical_set = set(critical_containers)
            
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Function to plot clusters on a given axis
            def plot_clusters(ax, clusters, title):
                # Create a colormap with enough colors for all clusters
                num_clusters = len(clusters)
                if num_clusters == 0:
                    return
                    
                # Use a qualitative colormap for discrete colors
                cmap = plt.cm.get_cmap('tab20', num_clusters)
                
                # Create a mapping from cluster IDs to indices for coloring
                cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(clusters.keys())}
                
                # Legend handles
                legend_handles = []
                
                # Plot each cluster
                for cluster_id, container_ids in clusters.items():
                    # Get valid containers with features
                    valid_containers = [c_id for c_id in container_ids if c_id in container_features]
                    if not valid_containers:
                        continue
                        
                    # Split into critical and non-critical
                    critical_containers = [c_id for c_id in valid_containers if c_id in critical_set]
                    non_critical_containers = [c_id for c_id in valid_containers if c_id not in critical_set]
                    
                    color = cmap(cluster_id_to_idx[cluster_id])
                    
                    # Plot non-critical with circles
                    if non_critical_containers:
                        non_critical_centroids = [container_features[c_id][0] for c_id in non_critical_containers]
                        non_critical_spans = [container_features[c_id][1] for c_id in non_critical_containers]
                        
                        ax.scatter(
                            non_critical_centroids,
                            non_critical_spans,
                            color=color,
                            marker='o',
                            s=50,
                            alpha=0.7,
                            edgecolors='black',
                            linewidths=0.5
                        )
                    
                    # Plot critical with X markers
                    if critical_containers:
                        critical_centroids = [container_features[c_id][0] for c_id in critical_containers]
                        critical_spans = [container_features[c_id][1] for c_id in critical_containers]
                        
                        ax.scatter(
                            critical_centroids,
                            critical_spans,
                            color=color,
                            marker='X',
                            s=100,
                            alpha=0.9,
                            edgecolors='black',
                            linewidths=1.0
                        )
                    
                    # Calculate cluster center for annotation
                    all_centroids = [container_features[c_id][0] for c_id in valid_containers]
                    all_spans = [container_features[c_id][1] for c_id in valid_containers]
                    
                    avg_centroid = sum(all_centroids) / len(all_centroids)
                    avg_span = sum(all_spans) / len(all_spans)
                    
                    # Annotate with cluster ID and counts
                    ax.annotate(
                        f"{cluster_id}: {len(critical_containers)}/{len(valid_containers)}",
                        xy=(avg_centroid, avg_span),
                        xytext=(0, 5),
                        textcoords="offset points",
                        fontsize=10,
                        weight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                    
                    # Add to legend
                    legend_handles.append(
                        mpatches.Patch(color=color, label=f'Cluster {cluster_id}')
                    )
                
                # Add marker legend items
                legend_handles.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=10, label='Non-Critical')
                )
                legend_handles.append(
                    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', 
                           markersize=10, label='Critical')
                )
                
                # Set title and labels
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Aisle Centroid', fontsize=12)
                ax.set_ylabel('Aisle Span', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add legend if not too many clusters
                if num_clusters <= 12:  # Limit for readability
                    ax.legend(
                        handles=legend_handles,
                        loc='upper center', 
                        bbox_to_anchor=(0.5, -0.1),
                        ncol=min(4, num_clusters + 2),
                        fontsize=9
                    )
            
            # Plot before and after
            plot_clusters(ax1, before_clusters, "Before Augmentation")
            plot_clusters(ax2, after_clusters, "After Augmentation")
            
            # Add overall title
            fig.suptitle("Cluster Comparison: Before and After Augmentation", fontsize=16)
            
            # Save figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.timing_stats['visualize_cluster_comparison'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing cluster comparison: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def visualize_additional_clusters(self, 
                                    clusters: Dict[str, List[str]],
                                    container_features: Dict[str, Tuple[float, float, int]],
                                    is_selected: bool = False,
                                    output_path: str = None) -> None:
        """
        Visualize additional clusters formed from remaining containers.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        is_selected : bool, optional
            Whether these are the selected clusters (post-selection) or all clusters (pre-selection)
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        start_time = time.time()
        
        if not self.enabled:
            return
            
        if not clusters:
            self.logger.warning("No clusters to visualize for additional clusters")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                stage = 'selected' if is_selected else 'formed'
                output_path = os.path.join(self.additional_clusters_path, f'additional_clusters_{stage}.png')
                
            self.logger.debug(f"Generating additional clusters visualization ({stage}): {output_path}")
            
            # Create the figure with adjusted size based on cluster count
            num_clusters = len(clusters)
            plt.figure(figsize=(16, 10))
            
            # Create a colormap with enough colors for all clusters
            # For many clusters, use a continuous colormap to avoid too many similar colors
            if num_clusters <= 20:
                cmap = plt.cm.get_cmap('tab20', num_clusters)
            else:
                cmap = plt.cm.get_cmap('viridis', num_clusters)
            
            # Create a mapping from original cluster IDs to sequential integers for coloring
            cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(clusters.keys())}
            
            # Prepare data for scatter plot
            centroids = []
            spans = []
            colors = []
            cluster_ids = []  # Store cluster ID for each point for annotation
            
            for cluster_id, container_ids in clusters.items():
                for container_id in container_ids:
                    if container_id in container_features:
                        centroid, span, _ = container_features[container_id]
                        centroids.append(centroid)
                        spans.append(span)
                        colors.append(cmap(cluster_id_to_idx[cluster_id]))
                        cluster_ids.append(cluster_id)
            
            # Skip if no valid containers to plot
            if not centroids:
                self.logger.warning("No valid containers with features for visualization")
                return
            
            # Plot the scatter points
            scatter = plt.scatter(centroids, spans, c=colors, alpha=0.7, s=50)
            
            # Add title and labels
            stage_text = "Selected" if is_selected else "Formed"
            plt.title(f'Additional Clusters ({stage_text}): {num_clusters} clusters', fontsize=14)
            plt.xlabel('Aisle Centroid', fontsize=12)
            plt.ylabel('Aisle Span', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # For larger datasets, add callout to indicate cluster count
            if num_clusters > 20:
                plt.figtext(0.5, 0.01, 
                          f"{num_clusters} clusters visualized (legend omitted due to size)",
                          ha="center", fontsize=10, 
                          bbox={"boxstyle":"round", "alpha":0.1})
            
            # Add a legend only for small number of clusters
            if num_clusters <= 20:
                # For small number of clusters, show full legend
                legend_elements = [
                    mpatches.Patch(facecolor=cmap(cluster_id_to_idx[cluster_id]), 
                                 edgecolor='black',
                                 alpha=0.7,
                                 label=f'Cluster {cluster_id}')
                    for cluster_id in clusters.keys()
                ]
                
                plt.legend(
                    handles=legend_elements, 
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    fontsize=9
                )
            
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for text
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Generate cluster statistics visualization
            stats_output_path = os.path.join(self.stats_path, f'additional_cluster_stats_{stage}.png')
            
            self._visualize_cluster_sizes(
                clusters,
                container_features,
                stats_output_path
            )
            
            self.timing_stats['visualize_additional_clusters'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing additional clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _visualize_cluster_sizes(self, 
                              clusters: Dict[str, List[str]],
                              container_features: Dict[str, Tuple[float, float, int]],
                              output_path: str) -> None:
        """
        Generate a bar chart visualization of cluster sizes.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        output_path : str
            Path to save visualization
        """
        try:
            # Collect statistics by cluster
            cluster_sizes = []
            cluster_avg_centroids = []
            cluster_avg_spans = []
            cluster_labels = []
            
            for cluster_id, container_ids in clusters.items():
                valid_containers = [c_id for c_id in container_ids if c_id in container_features]
                
                if valid_containers:
                    cluster_sizes.append(len(valid_containers))
                    cluster_labels.append(cluster_id)
                    
                    # Calculate statistics
                    centroids = [container_features[c_id][0] for c_id in valid_containers]
                    spans = [container_features[c_id][1] for c_id in valid_containers]
                    
                    cluster_avg_centroids.append(np.mean(centroids))
                    cluster_avg_spans.append(np.mean(spans))
            
            # Sort everything by cluster size
            sorted_indices = np.argsort(cluster_sizes)[::-1]  # Descending order
            sorted_sizes = [cluster_sizes[i] for i in sorted_indices]
            sorted_labels = [cluster_labels[i] for i in sorted_indices]
            
            # Create figure with adjusted height for many clusters
            num_clusters = len(sorted_labels)
            # Scale height based on number of clusters (min 8, max 24)
            fig_height = min(max(8, num_clusters * 0.3), 24)
            plt.figure(figsize=(14, fig_height))
            
            # Plot horizontal bars instead of vertical for better readability with many clusters
            bars = plt.barh(range(num_clusters), sorted_sizes, color='skyblue')
            plt.ylabel('Cluster', fontsize=12)
            plt.xlabel('Number of Containers', fontsize=12)
            plt.title(f'Cluster Size Distribution ({num_clusters} clusters)', fontsize=14)
            
            # Adjust font size for cluster labels based on count
            label_fontsize = max(5, min(9, 300 / num_clusters))
            plt.yticks(range(num_clusters), sorted_labels, fontsize=label_fontsize)
            plt.grid(True, axis='x', alpha=0.3)
            
            # Adding the values at the end of each bar
            # Only add text if there aren't too many clusters
            if num_clusters <= 50:
                for i, v in enumerate(sorted_sizes):
                    plt.text(v + 0.5, i, str(v), va='center', fontsize=label_fontsize)
            
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout(pad=2.0)  # Add extra padding
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating cluster size visualization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def visualize_final_clusters(self, 
                               clusters: Dict[str, List[str]],
                               critical_containers: List[str],
                               container_features: Dict[str, Tuple[float, float, int]],
                               output_path: str = None) -> None:
        """
        Visualize final cluster assignments.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary mapping cluster IDs to lists of container IDs
        critical_containers : List[str]
            List of critical container IDs
        container_features : Dict[str, Tuple[float, float, int]]
            Dictionary mapping container IDs to feature tuples
        output_path : str, optional
            Path to save visualization, by default None (uses default path)
        """
        start_time = time.time()
        
        if not self.enabled:
            return
            
        if not clusters:
            self.logger.warning("No clusters to visualize for final clusters")
            return
            
        try:
            # Use provided output path or default
            if output_path is None:
                output_path = os.path.join(self.final_clusters_path, 'final_clusters.png')
                
            self.logger.debug(f"Generating final clusters visualization: {output_path}")
            
            # Create a set of critical containers for faster lookups
            critical_set = set(critical_containers)
            
            # Create the figure
            plt.figure(figsize=(16, 10))
            
            # Create a colormap with enough colors for all clusters
            num_clusters = len(clusters)
            cmap = plt.cm.get_cmap('tab20', num_clusters)
            
            # Create a mapping from cluster IDs to indices for coloring
            cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(clusters.keys())}
            
            # Prepare data per cluster for legend and scatter plot
            cluster_data = {}
            
            for cluster_id, container_ids in clusters.items():
                cluster_idx = cluster_id_to_idx[cluster_id]
                color = cmap(cluster_idx)
                
                # Initialize data structures for this cluster
                cluster_data[cluster_id] = {
                    'centroids': [],
                    'spans': [],
                    'is_critical': [],
                    'color': color
                }
                
                # Process containers
                for container_id in container_ids:
                    if container_id in container_features:
                        centroid, span, _ = container_features[container_id]
                        is_critical = container_id in critical_set
                        
                        cluster_data[cluster_id]['centroids'].append(centroid)
                        cluster_data[cluster_id]['spans'].append(span)
                        cluster_data[cluster_id]['is_critical'].append(is_critical)
            
            # Create custom legend handles
            legend_handles = []
            
            # Plot each cluster separately
            for cluster_id, data in cluster_data.items():
                # Skip clusters with no valid data
                if not data['centroids']:
                    continue
                    
                # Plot non-critical containers with 'o' marker
                non_critical_mask = [not x for x in data['is_critical']]
                if any(non_critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if non_critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if non_critical_mask[i]],
                        color=data['color'],
                        marker='o',
                        s=50,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5
                    )
                
                # Plot critical containers with 'X' marker
                critical_mask = data['is_critical']
                if any(critical_mask):
                    plt.scatter(
                        [data['centroids'][i] for i in range(len(data['centroids'])) if critical_mask[i]],
                        [data['spans'][i] for i in range(len(data['spans'])) if critical_mask[i]],
                        color=data['color'],
                        marker='X',
                        s=100,
                        alpha=0.9,
                        edgecolors='black',
                        linewidths=1.0
                    )
                
                # Create legend handle for this cluster
                legend_handles.append(
                    mpatches.Patch(color=data['color'], label=f'Cluster {cluster_id}')
                )
                
                # Calculate cluster centroid for annotation
                avg_centroid = sum(data['centroids']) / len(data['centroids'])
                avg_span = sum(data['spans']) / len(data['spans'])
                
                # Count critical containers in this cluster
                critical_count = sum(data['is_critical'])
                total_count = len(data['centroids'])
                
                # Annotate cluster ID and stats at cluster centroid
                plt.annotate(
                    f"{cluster_id}: {critical_count}/{total_count}",
                    xy=(avg_centroid, avg_span),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=10,
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Add markers to legend for critical and non-critical
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=10, label='Non-Critical')
            )
            legend_handles.append(
                Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', 
                       markersize=10, label='Critical')
            )
            
            # Add labels and title
            plt.xlabel('Aisle Centroid', fontsize=12)
            plt.ylabel('Aisle Span', fontsize=12)
            plt.title('Final Cluster Assignments', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add legend in two parts - clusters and markers
            if num_clusters <= 20:  # Only show legend for a reasonable number of clusters
                plt.legend(
                    handles=legend_handles,
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=min(5, num_clusters + 2),  # Distribute in columns
                    frameon=True,
                    fontsize=10
                )
            
            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Generate statistics visualization
            stats_output_path = os.path.join(self.stats_path, 'final_cluster_stats.png')
            self.visualize_cluster_stats(
                clusters, 
                critical_containers,
                container_features,
                stats_output_path
            )
            
            self.timing_stats['visualize_final_clusters'] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Error visualizing final clusters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())