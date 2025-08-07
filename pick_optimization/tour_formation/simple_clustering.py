"""
Simplified Linear Clustering Script

This script provides a simplified, linear approach to container clustering
with minimal use of functions and classes. It consolidates all the complex
clustering logic from the original modular system into a single, easy-to-understand
and modify script.

The script handles:
1. Feature extraction from container and slotbook data
2. Critical container identification and prioritization
3. Hierarchical clustering with seed centers
4. Cluster augmentation and optimization
5. Tour calculation and final cluster assignment
"""

import pandas as pd
import numpy as np
import logging
import time
import math
from typing import Dict, List, Tuple, Any
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from datetime import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_cluster_containers(container_data: pd.DataFrame, 
                             slotbook_data: pd.DataFrame,
                             max_cluster_size: int = 200,
                             containers_per_tour: int = 50,
                             max_picking_capacity: int = 1000,
                             linkage_method: str = 'ward',
                             generate_visualizations: bool = False) -> Dict[str, List[str]]:
    """
    Main clustering function - simplified linear approach.
    
    Parameters:
    -----------
    container_data : pd.DataFrame
        DataFrame with columns: container_id, item_number, slack_category (optional)
    slotbook_data : pd.DataFrame  
        DataFrame with columns: item_number, aisle_sequence
    max_cluster_size : int
        Maximum containers per cluster
    containers_per_tour : int
        Maximum containers per tour
    max_picking_capacity : int
        Maximum total containers to process
    linkage_method : str
        Hierarchical clustering linkage method
    generate_visualizations : bool
        Whether to generate cluster visualizations
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    
    start_time = time.time()
    logger.info("Starting simplified container clustering")
    
    # === STEP 1: DATA VALIDATION AND PREPARATION ===
    if container_data.empty or slotbook_data.empty:
        logger.error("Empty input data provided")
        return {}
    
    container_ids = container_data['container_id'].unique().tolist()
    logger.info(f"Processing {len(container_ids)} unique containers")
    
    # Early return for small datasets
    if len(container_ids) <= max_cluster_size:
        logger.info(f"All containers fit in single cluster")
        return {'1': container_ids}
    
    # === STEP 2: BUILD SKU-AISLE MAPPING ===
    logger.info("Building SKU-aisle mapping")
    sku_aisle_mapping = {}
    for sku, group in slotbook_data.groupby('item_number'):
        aisles = sorted(group['aisle_sequence'].unique().tolist())
        sku_aisle_mapping[sku] = aisles
    
    # === STEP 3: EXTRACT CONTAINER FEATURES ===
    logger.info("Extracting container features")
    container_features = {}
    
    for container_id in container_ids:
        # Get SKUs for this container
        container_skus = container_data[container_data['container_id'] == container_id]['item_number'].unique()
        
        # Get optimized aisles for this container
        container_aisles = set()
        for sku in container_skus:
            if sku in sku_aisle_mapping:
                aisles = sku_aisle_mapping[sku]
                if len(aisles) == 1:
                    # Single location SKU - must visit
                    container_aisles.add(aisles[0])
                else:
                    # Multi-location SKU - choose closest to existing aisles
                    if not container_aisles:
                        container_aisles.add(aisles[0])
                    else:
                        # Find aisle closest to existing ones
                        best_aisle = min(aisles, key=lambda a: min(abs(a - existing) for existing in container_aisles))
                        container_aisles.add(best_aisle)
        
        if container_aisles:
            # Calculate features: centroid, span, distinct_aisles
            aisles_list = list(container_aisles)
            centroid = sum(aisles_list) / len(aisles_list)
            span = max(aisles_list) - min(aisles_list) if len(aisles_list) > 1 else 0
            distinct_aisles = len(aisles_list)
            container_features[container_id] = (centroid, span, distinct_aisles)
    
    logger.info(f"Extracted features for {len(container_features)} containers")
    
    # === STEP 4: IDENTIFY CRITICAL CONTAINERS ===
    critical_containers = []
    if 'slack_category' in container_data.columns:
        critical_df = container_data[['container_id', 'slack_category']].drop_duplicates()
        critical_containers = critical_df[
            critical_df['slack_category'].isin(['Critical', 'Urgent'])
        ]['container_id'].unique().tolist()
        logger.info(f"Found {len(critical_containers)} critical containers")
    
    # === STEP 5: DECIDE CLUSTERING STRATEGY ===
    if not critical_containers or len(container_ids) <= max_picking_capacity:
        # Standard clustering path - no critical containers or all fit in capacity
        logger.info("Using standard clustering path")
        clusters = _standard_clustering_path(container_ids, container_features, max_cluster_size, linkage_method)
    else:
        # Critical container prioritization path
        logger.info("Using critical container prioritization path")
        clusters = _critical_container_path(container_ids, critical_containers, container_features, 
                                          max_cluster_size, max_picking_capacity, containers_per_tour, linkage_method)
    
    # === STEP 6: FINALIZE AND RENUMBER CLUSTERS ===
    final_clusters = _finalize_clusters(clusters, container_features, critical_containers, containers_per_tour)
    
    # === STEP 7: LOG RESULTS ===
    total_time = time.time() - start_time
    total_containers = sum(len(c) for c in final_clusters.values())
    total_tours = sum(math.ceil(len(c) / containers_per_tour) for c in final_clusters.values())
    
    logger.info(f"Clustering completed in {total_time:.2f} seconds")
    logger.info(f"Formed {len(final_clusters)} clusters with {total_containers} containers")
    logger.info(f"Total tours required: {total_tours}")
    
    # Print cluster statistics
    for cluster_id, containers in final_clusters.items():
        critical_count = sum(1 for c in containers if c in critical_containers)
        tours = math.ceil(len(containers) / containers_per_tour)
        logger.info(f"Cluster {cluster_id}: {len(containers)} containers ({critical_count} critical) - {tours} tours")
    
    return final_clusters


def _standard_clustering_path(container_ids: List[str], 
                             container_features: Dict[str, Tuple[float, float, int]],
                             max_cluster_size: int,
                             linkage_method: str) -> Dict[str, List[str]]:
    """Standard clustering when no critical containers or all fit in capacity."""
    
    # Prepare feature matrix
    feature_arrays = []
    valid_containers = []
    
    for c_id in container_ids:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if not valid_containers:
        return {}
    
    # Normalize features
    feature_matrix = np.array(feature_arrays)
    feature_matrix = (feature_matrix - feature_matrix.min(axis=0)) / (feature_matrix.max(axis=0) - feature_matrix.min(axis=0) + 1e-8)
    
    # Determine optimal number of clusters
    estimated_clusters = max(2, len(valid_containers) // max_cluster_size + 1)
    max_possible = min(10, len(valid_containers) // 2)
    optimal_clusters = min(estimated_clusters, max_possible)
    
    # Perform hierarchical clustering
    Z = linkage(feature_matrix, method=linkage_method)
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to clusters
    clusters = {}
    for i, container_id in enumerate(valid_containers):
        cluster_id = int(cluster_labels[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(container_id)
    
    # Break down oversized clusters
    final_clusters = {}
    cluster_counter = 1
    
    for cluster_id, containers in clusters.items():
        if len(containers) <= max_cluster_size:
            final_clusters[str(cluster_counter)] = containers
            cluster_counter += 1
        else:
            # Subdivide oversized cluster
            sub_clusters = _subdivide_cluster(containers, container_features, max_cluster_size, linkage_method)
            for sub_containers in sub_clusters.values():
                final_clusters[str(cluster_counter)] = sub_containers
                cluster_counter += 1
    
    return final_clusters


def _critical_container_path(container_ids: List[str],
                            critical_containers: List[str],
                            container_features: Dict[str, Tuple[float, float, int]],
                            max_cluster_size: int,
                            max_picking_capacity: int,
                            containers_per_tour: int,
                            linkage_method: str) -> Dict[str, List[str]]:
    """Critical container prioritization clustering path."""
    
    # Separate containers
    non_critical_containers = [c_id for c_id in container_ids if c_id not in critical_containers]
    
    # If critical containers exceed capacity, prioritize them only
    if len(critical_containers) > max_picking_capacity:
        non_critical_containers = []
    
    # === STEP 1: Form seed clusters from critical containers ===
    seed_clusters = _form_seed_clusters(critical_containers, container_features, max_cluster_size, linkage_method)
    
    if not seed_clusters:
        logger.error("Failed to form seed clusters")
        return {}
    
    # === STEP 2: Calculate cluster centers ===
    cluster_centers = {}
    for cluster_id, containers in seed_clusters.items():
        centroids = [container_features[c_id][0] for c_id in containers if c_id in container_features]
        spans = [container_features[c_id][1] for c_id in containers if c_id in container_features]
        if centroids:
            cluster_centers[cluster_id] = (np.mean(centroids), np.mean(spans))
    
    # === STEP 3: Assign non-critical containers to nearest seed centers ===
    kmeans_clusters = {cid: [] for cid in seed_clusters.keys()}
    
    for c_id in non_critical_containers:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            # Find closest cluster center
            min_distance = float('inf')
            closest_cluster = None
            
            for cluster_id, center in cluster_centers.items():
                distance = np.sqrt((centroid - center[0])**2 + (span - center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            if closest_cluster:
                kmeans_clusters[closest_cluster].append(c_id)
    
    # === STEP 4: Merge and augment clusters ===
    merged_clusters = {}
    for cluster_id in seed_clusters.keys():
        critical_in_cluster = seed_clusters[cluster_id]
        non_critical_in_cluster = kmeans_clusters.get(cluster_id, [])
        merged_clusters[cluster_id] = critical_in_cluster + non_critical_in_cluster
    
    # Remove assigned non-critical containers from remaining pool
    assigned_non_critical = set()
    for containers in kmeans_clusters.values():
        assigned_non_critical.update(containers)
    remaining_non_critical = list(set(non_critical_containers) - assigned_non_critical)
    
    # Augment clusters with remaining non-critical containers
    augmented_clusters, remaining_containers = _augment_clusters(
        merged_clusters, remaining_non_critical, critical_containers, 
        container_features, max_cluster_size
    )
    
    # === STEP 5: Handle remaining capacity with additional clusters ===
    total_seed_tours = sum(math.ceil(len(c) / containers_per_tour) for c in augmented_clusters.values())
    remaining_capacity = (max_picking_capacity // containers_per_tour) - total_seed_tours
    
    final_clusters = augmented_clusters.copy()
    
    if remaining_capacity > 0 and remaining_containers:
        # Form additional clusters from remaining containers
        additional_clusters = _standard_clustering_path(remaining_containers, container_features, max_cluster_size, linkage_method)
        
        # Select best clusters to fill remaining capacity
        selected_additional = _select_additional_clusters(
            additional_clusters, container_features, remaining_capacity, containers_per_tour
        )
        
        # Add selected clusters to final result
        next_id = len(final_clusters) + 1
        for containers in selected_additional.values():
            final_clusters[str(next_id)] = containers
            next_id += 1
    
    return final_clusters


def _form_seed_clusters(critical_containers: List[str],
                       container_features: Dict[str, Tuple[float, float, int]],
                       max_cluster_size: int,
                       linkage_method: str) -> Dict[int, List[str]]:
    """Form seed clusters from critical containers."""
    
    # Prepare feature matrix
    feature_arrays = []
    valid_containers = []
    
    for c_id in critical_containers:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if len(valid_containers) < 2:
        return {0: valid_containers} if valid_containers else {}
    
    # Normalize features
    feature_matrix = np.array(feature_arrays)
    feature_matrix = (feature_matrix - feature_matrix.min(axis=0)) / (feature_matrix.max(axis=0) - feature_matrix.min(axis=0) + 1e-8)
    
    # Determine optimal number of clusters
    min_clusters = max(2, int(len(valid_containers) / max_cluster_size))
    max_clusters = min(10, len(valid_containers) // 2)
    optimal_clusters = min_clusters
    
    # Try different numbers of clusters and pick the best
    best_score = -1
    best_clusters = None
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        Z = linkage(feature_matrix, method=linkage_method)
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        try:
            score = silhouette_score(feature_matrix, cluster_labels)
            if score > best_score:
                best_score = score
                best_clusters = cluster_labels
        except:
            continue
    
    if best_clusters is None:
        best_clusters = fcluster(linkage(feature_matrix, method=linkage_method), min_clusters, criterion='maxclust')
    
    # Map containers to clusters
    clusters = {}
    for i, container_id in enumerate(valid_containers):
        cluster_id = int(best_clusters[i]) - 1  # Convert to 0-based
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(container_id)
    
    # Break down oversized clusters
    final_clusters = {}
    cluster_counter = 0
    
    for cluster_id, containers in clusters.items():
        if len(containers) <= max_cluster_size:
            final_clusters[cluster_counter] = containers
            cluster_counter += 1
        else:
            # Subdivide oversized cluster
            sub_clusters = _subdivide_cluster(containers, container_features, max_cluster_size, linkage_method)
            for sub_containers in sub_clusters.values():
                final_clusters[cluster_counter] = sub_containers
                cluster_counter += 1
    
    return final_clusters


def _subdivide_cluster(containers: List[str],
                      container_features: Dict[str, Tuple[float, float, int]],
                      max_cluster_size: int,
                      linkage_method: str,
                      max_depth: int = 3,
                      depth: int = 0) -> Dict[int, List[str]]:
    """Recursively subdivide oversized clusters."""
    
    if depth >= max_depth or len(containers) <= 1:
        return {0: containers}
    
    # Prepare features
    feature_arrays = []
    valid_containers = []
    
    for c_id in containers:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if len(valid_containers) <= 1:
        return {0: valid_containers}
    
    # Normalize and cluster
    feature_matrix = np.array(feature_arrays)
    feature_matrix = (feature_matrix - feature_matrix.min(axis=0)) / (feature_matrix.max(axis=0) - feature_matrix.min(axis=0) + 1e-8)
    
    Z = linkage(feature_matrix, method=linkage_method)
    cluster_labels = fcluster(Z, 2, criterion='maxclust')  # Always split into 2
    
    # Map to subclusters
    sub_clusters = {}
    for i, container_id in enumerate(valid_containers):
        sub_id = int(cluster_labels[i]) - 1
        if sub_id not in sub_clusters:
            sub_clusters[sub_id] = []
        sub_clusters[sub_id].append(container_id)
    
    # Recursively subdivide if needed
    final_sub_clusters = {}
    cluster_counter = 0
    
    for sub_id, sub_containers in sub_clusters.items():
        if len(sub_containers) <= max_cluster_size:
            final_sub_clusters[cluster_counter] = sub_containers
            cluster_counter += 1
        else:
            # Recursively subdivide
            recursive_clusters = _subdivide_cluster(
                sub_containers, container_features, max_cluster_size, 
                linkage_method, max_depth, depth + 1
            )
            for recursive_containers in recursive_clusters.values():
                final_sub_clusters[cluster_counter] = recursive_containers
                cluster_counter += 1
    
    return final_sub_clusters


def _augment_clusters(seed_clusters: Dict[int, List[str]],
                     remaining_containers: List[str],
                     critical_containers: List[str],
                     container_features: Dict[str, Tuple[float, float, int]],
                     max_cluster_size: int) -> Tuple[Dict[int, List[str]], List[str]]:
    """Augment seed clusters with remaining non-critical containers."""
    
    critical_set = set(critical_containers)
    final_clusters = {k: v.copy() for k, v in seed_clusters.items()}
    remaining = set(remaining_containers)
    
    # Calculate cluster centers
    cluster_centers = {}
    for cluster_id, containers in seed_clusters.items():
        centroids = [container_features[c_id][0] for c_id in containers if c_id in container_features]
        spans = [container_features[c_id][1] for c_id in containers if c_id in container_features]
        if centroids:
            cluster_centers[cluster_id] = (np.mean(centroids), np.mean(spans))
    
    # For each cluster, add non-critical containers up to capacity
    for cluster_id, containers in final_clusters.items():
        # Count critical containers
        critical_count = sum(1 for c_id in containers if c_id in critical_set)
        space_left = max_cluster_size - critical_count
        
        if space_left > 0:
            # Calculate distances to cluster center
            center = cluster_centers.get(cluster_id, (0, 0))
            container_distances = []
            
            for c_id in remaining:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    distance = np.sqrt((centroid - center[0])**2 + (span - center[1])**2)
                    container_distances.append((c_id, distance))
            
            # Sort by distance and add closest containers
            container_distances.sort(key=lambda x: x[1])
            containers_to_add = [c_id for c_id, _ in container_distances[:space_left]]
            
            final_clusters[cluster_id].extend(containers_to_add)
            
            # Remove from remaining pool
            for c_id in containers_to_add:
                if c_id in remaining:
                    remaining.remove(c_id)
    
    return final_clusters, list(remaining)


def _select_additional_clusters(additional_clusters: Dict[str, List[str]],
                               container_features: Dict[str, Tuple[float, float, int]],
                               remaining_capacity: int,
                               containers_per_tour: int) -> Dict[str, List[str]]:
    """Select best additional clusters to fill remaining capacity."""
    
    if remaining_capacity <= 0 or not additional_clusters:
        return {}
    
    # Calculate quality and tours for each cluster
    cluster_metrics = {}
    for cluster_id, containers in additional_clusters.items():
        # Simple quality metric based on spatial cohesion
        centroids = [container_features[c_id][0] for c_id in containers if c_id in container_features]
        spans = [container_features[c_id][1] for c_id in containers if c_id in container_features]
        
        if centroids:
            centroid_std = np.std(centroids) if len(centroids) > 1 else 0
            span_std = np.std(spans) if len(spans) > 1 else 0
            quality = 1.0 / (centroid_std + span_std + 0.1)  # Higher is better
        else:
            quality = 0
        
        tours = max(1, len(containers) // containers_per_tour)
        
        cluster_metrics[cluster_id] = {
            'containers': containers,
            'quality': quality,
            'tours': tours
        }
    
    # Sort by quality and select greedily
    sorted_clusters = sorted(cluster_metrics.items(), key=lambda x: x[1]['quality'], reverse=True)
    
    selected = {}
    total_tours = 0
    
    for cluster_id, info in sorted_clusters:
        selected[cluster_id] = info['containers']
        total_tours += info['tours']
        
        if total_tours >= remaining_capacity:
            break
    
    return selected


def _finalize_clusters(clusters: Dict[str, List[str]],
                      container_features: Dict[str, Tuple[float, float, int]],
                      critical_containers: List[str],
                      containers_per_tour: int) -> Dict[str, List[str]]:
    """Finalize clusters with sequential IDs and calculate tours."""
    
    # Sort clusters by minimum aisle for logical ordering
    sorted_clusters = []
    for cluster_id, containers in clusters.items():
        min_aisle = float('inf')
        for c_id in containers:
            if c_id in container_features:
                centroid, _, _ = container_features[c_id]
                min_aisle = min(min_aisle, centroid)
        
        if min_aisle == float('inf'):
            min_aisle = 9999
        
        sorted_clusters.append((cluster_id, containers, min_aisle))
    
    # Sort by minimum aisle and renumber
    sorted_clusters.sort(key=lambda x: x[2])
    final_clusters = {}
    
    for idx, (_, containers, _) in enumerate(sorted_clusters, 1):
        final_clusters[str(idx)] = containers
    
    return final_clusters


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Example usage with sample data
    logger.info("Running example clustering")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample container data
    container_data = pd.DataFrame({
        'container_id': [f'C{i:03d}' for i in range(1, 101)],
        'item_number': [f'SKU{i:03d}' for i in range(1, 101)],
        'slack_category': np.random.choice(['Normal', 'Critical', 'Urgent'], 100, p=[0.7, 0.2, 0.1])
    })
    
    # Sample slotbook data
    slotbook_data = pd.DataFrame({
        'item_number': [f'SKU{i:03d}' for i in range(1, 101)],
        'aisle_sequence': np.random.randint(1, 51, 100)
    })
    
    # Run clustering
    clusters = simple_cluster_containers(
        container_data=container_data,
        slotbook_data=slotbook_data,
        max_cluster_size=20,
        containers_per_tour=10,
        max_picking_capacity=500
    )
    
    logger.info("Example clustering completed") 