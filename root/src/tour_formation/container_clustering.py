import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from scipy.cluster.hierarchy import linkage, fcluster
import logging
import time
import os
import matplotlib.pyplot as plt
import math

logger = logging.getLogger(__name__)

def get_critical_containers(container_data: pd.DataFrame, container_ids: List[str]) -> List[str]:
    """
    Extract critical and urgent containers from container data.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with slack_category column if available
    container_ids : List[str]
        List of container IDs to filter from
        
    Returns
    -------
    List[str]
        List of critical and urgent container IDs
    """
    # Check if slack data is available
    has_slack_data = 'slack_category' in container_data.columns
    
    if not has_slack_data:
        logger.info("No slack data available, no critical containers identified")
        return []  # No critical containers if no slack data
    
    # Get containers with 'Critical' or 'Urgent' status
    critical_df = container_data[container_data['slack_category'].isin(['Critical', 'Urgent'])]
    critical_container_ids = critical_df['container_id'].unique().tolist()
    
    # Filter to only include container_ids that are in our input list
    result = [c_id for c_id in critical_container_ids if c_id in container_ids]
    
    logger.info(f"Found {len(result)} critical/urgent containers out of {len(container_ids)} total containers")
    return result

def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0,1] range for each column.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix of features to normalize
        
    Returns
    -------
    np.ndarray
        Normalized feature matrix
    """
    if feature_matrix.size == 0 or feature_matrix.shape[0] == 0:
        return feature_matrix
        
    result = feature_matrix.copy()
    
    # Normalize each column separately
    for col in range(feature_matrix.shape[1]):
        col_max = np.max(feature_matrix[:, col])
        col_min = np.min(feature_matrix[:, col])
        range_val = col_max - col_min
        
        if range_val > 0:
            result[:, col] = (feature_matrix[:, col] - col_min) / range_val
        else:
            # If all values are the same, set to 0.5
            result[:, col] = 0.5
    
    return result

def compute_container_features(container_id: str, container_data: pd.DataFrame, 
                              sku_aisle_mapping: Dict[str, List[int]]) -> Tuple[float, float, int]:
    """
    Compute feature vector for a container: (aisle_centroid, aisle_span, distinct_aisles)
    
    Parameters
    ----------
    container_id : str
        ID of the container
    container_data : pd.DataFrame
        Container data with SKU information
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
        
    Returns
    -------
    Tuple[float, float, int]
        Tuple containing (aisle_centroid, aisle_span, distinct_aisles)
    """
    # Get container aisles
    aisles = get_container_aisles(container_id, container_data, sku_aisle_mapping)
    
    if not aisles:
        return 0, 0, 0
    
    #dispersion = calculate_aisle_dispersion(aisles)
    
    # Calculate aisle centroid and span
    centroid = sum(aisles) / len(aisles)
    span = max(aisles) - min(aisles) if len(aisles) > 1 else 0
    
    # Count distinct aisles
    distinct_aisles = len(aisles)
    
    return centroid, span, distinct_aisles

def get_container_aisles(container_id: str, container_data: pd.DataFrame, 
                        sku_aisle_mapping: Dict[str, List[int]]) -> Set[int]:
    """
    Get optimized aisles required for a specific container, minimizing total aisles visited.
    Chooses the best aisle for multi-location SKUs by considering other SKUs in the container.
    """
    # Get all SKUs for this container
    container_skus = container_data[container_data['container_id'] == container_id]['item_number'].unique()

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

def calculate_aisle_dispersion(aisles):
    """Calculate standard deviation of aisle numbers"""
    if len(aisles) <= 1:
        return 0
    return np.std(list(aisles))


def extract_container_features(container_data: pd.DataFrame, 
                              sku_aisle_mapping: Dict[str, List[int]],
                              container_ids: Optional[List[str]] = None) -> Dict[str, Tuple[float, float, int]]:
    """
    Extract features (centroid, span, distinct_aisles) for all containers.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with SKU information
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
    container_ids : Optional[List[str]]
        Optional list of specific container IDs to process
        
    Returns
    -------
    Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    """
    if container_ids is None:
        container_ids = container_data['container_id'].unique().tolist()
    
    container_features = {}
    
    for container_id in container_ids:
        # Re-use existing function from container_clustering.py
        centroid, span, distinct_aisles = compute_container_features(
            container_id, container_data, sku_aisle_mapping
        )
        
        if centroid != 0 or span != 0:  # Skip containers with no valid aisle data
            container_features[container_id] = (centroid, span, distinct_aisles)
    
    logger.info(f"Extracted features for {len(container_features)} containers")
    return container_features

def calculate_cluster_centers(clusters: Dict[str, List[str]], 
                             container_features: Dict[str, Tuple[float, float, int]]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate center for each cluster based on container features.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
        
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping cluster IDs to centers (centroid, span)
    """
    centers = {}
    
    for cluster_id, container_ids in clusters.items():
        valid_containers = [c_id for c_id in container_ids if c_id in container_features]
        
        if not valid_containers:
            logger.warning(f"Cluster {cluster_id} has no valid containers with features")
            continue
            
        # Calculate average centroid and span
        centroids = [container_features[c_id][0] for c_id in valid_containers]
        spans = [container_features[c_id][1] for c_id in valid_containers]
        
        avg_centroid = np.mean(centroids)
        avg_span = np.mean(spans)
        
        centers[cluster_id] = (avg_centroid, avg_span)
    
    logger.info(f"Calculated centers for {len(centers)} clusters")
    return centers

def form_seed_clusters(critical_container_ids: List[str], 
                      container_data: pd.DataFrame,
                      container_features: Dict[str, Tuple[float, float, int]],
                      max_cluster_size: int,
                      generate_visuals: bool = True,
                      output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Form initial seed clusters from critical containers using hierarchical clustering.
    
    Parameters
    ----------
    critical_container_ids : List[str]
        List of critical and urgent container IDs
    container_data : pd.DataFrame
        Container data with SKU information
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    max_cluster_size : int
        Maximum size for any cluster
    generate_visuals : bool, optional
        Whether to generate visualizations, by default True
    output_path : str, optional
        Path to save visualizations, by default './cluster_analysis'
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of critical container IDs
    """
    start_time = time.time()
    logger.info(f"Forming seed clusters from {len(critical_container_ids)} critical containers")
    
    # Create visualization output directory if needed
    if generate_visuals:
        os.makedirs(os.path.join(output_path, 'seed_clusters'), exist_ok=True)
    
    # Get feature matrix for critical containers
    critical_features = []
    valid_critical_containers = []
    
    for c_id in critical_container_ids:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            valid_critical_containers.append(c_id)
            
            # Use centroid and span as clustering features
            critical_features.append([centroid, span])
    
    if len(valid_critical_containers) < 2:
        # If only one valid critical container, return as a single cluster
        if len(valid_critical_containers) == 1:
            logger.warning("Only one valid critical container, returning as a single cluster")
            return {"0": valid_critical_containers}
        else:
            # This should never happen since we've already checked for critical containers
            logger.error("No valid critical containers found with features")
            raise ValueError("No valid critical containers with features")
    
    # Convert to numpy array
    feature_matrix = np.array(critical_features)
    
    # Normalize features
    normalized_features = normalize_features(feature_matrix)
    
    # Determine optimal number of clusters
    min_clusters = 2
    max_cluster_count = min(10, len(valid_critical_containers) // 2)
    
    # Ensure max_clusters is at least min_clusters
    if max_cluster_count < min_clusters:
        max_cluster_count = min_clusters
    
    optimal_clusters, silhouette_scores = determine_optimal_clusters(
        normalized_features, 
        min_clusters=min_clusters, 
        max_clusters=max_cluster_count,
        max_cluster_size = max_cluster_size   
    )
    
    logger.info(f"Determined optimal number of clusters: {optimal_clusters}")
    
    # Perform hierarchical clustering
    Z = linkage(normalized_features, method='ward')
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to clusters
    seed_clusters = {}
    for i, container_id in enumerate(valid_critical_containers):
        cluster_id = int(cluster_labels[i])
        cluster_id_str = str(cluster_id)  # Convert to string consistently
        
        if cluster_id_str not in seed_clusters:
            seed_clusters[cluster_id_str] = []
            
        seed_clusters[cluster_id_str].append(container_id)
    
    # Log cluster sizes
    for cluster_id, containers in seed_clusters.items():
        logger.info(f"Seed cluster {cluster_id}: {len(containers)} containers")
    
    # Apply sub-clustering if any cluster exceeds max_cluster_size
    final_seed_clusters = {}
    cluster_idx = 0
    
    for cluster_id, containers in seed_clusters.items():
        if len(containers) > max_cluster_size:
            logger.info(f"Seed cluster {cluster_id} exceeds max size, applying sub-clustering")
            
            # Re-cluster this group
            sub_feature_matrix = []
            sub_containers = []
            
            for c_id in containers:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    sub_feature_matrix.append([centroid, span])
                    sub_containers.append(c_id)
            
            # Skip if no valid containers
            if not sub_containers:
                continue
                
            # Convert to numpy array and normalize
            sub_features = normalize_features(np.array(sub_feature_matrix))
            
            # Calculate optimal sub-clusters
            sub_max_clusters = int(max(2, len(sub_containers) // max_cluster_size + 1))
            sub_optimal_clusters, _ = determine_optimal_clusters(
                sub_features, min_clusters=2, max_clusters=sub_max_clusters,
                max_cluster_size = max_cluster_size   
            )
            
            # Apply clustering
            sub_Z = linkage(sub_features, method='ward')
            sub_labels = fcluster(sub_Z, sub_optimal_clusters, criterion='maxclust')
            
            # Create sub-clusters
            for i, container_id in enumerate(sub_containers):
                sub_cluster_id = int(sub_labels[i])
                sub_key = f"{cluster_id}_{sub_cluster_id}"
                
                if sub_key not in final_seed_clusters:
                    final_seed_clusters[sub_key] = []
                    
                final_seed_clusters[sub_key].append(container_id)
        else:
            # Keep as is if size is acceptable
            final_seed_clusters[str(cluster_idx)] = containers
            cluster_idx += 1
    
    # Final check for any clusters still exceeding max_cluster_size
    overflow_clusters = {k: v for k, v in final_seed_clusters.items() if len(v) > max_cluster_size}
    
    if overflow_clusters:
        logger.warning(f"Found {len(overflow_clusters)} clusters still exceeding max_cluster_size")
        
        # Simple split for any remaining large clusters
        adjusted_clusters = {}
        next_id = max([int(k.split('_')[-1]) for k in final_seed_clusters.keys()], default=0) + 1
        
        for k, containers in final_seed_clusters.items():
            if len(containers) <= max_cluster_size:
                adjusted_clusters[k] = containers
            else:
                # Split into chunks
                chunks = [containers[i:i + max_cluster_size] 
                         for i in range(0, len(containers), max_cluster_size)]
                
                for i, chunk in enumerate(chunks):
                    adjusted_clusters[f"{k}_{next_id + i}"] = chunk
                
                next_id += len(chunks)
        
        final_seed_clusters = adjusted_clusters
    
    # Generate visualization for seed clusters if requested
    if generate_visuals:
        try:
            # Visualize the initial seed clusters
            visualize_seed_clusters(
                seed_clusters, 
                container_features, 
                normalized_features, 
                cluster_labels, 
                valid_critical_containers,
                os.path.join(output_path, 'seed_clusters', 'initial_seed_clusters.png')
            )
            
            # Visualize the final seed clusters after any sub-clustering
            visualize_seed_clusters(
                final_seed_clusters,
                container_features,
                output_path=os.path.join(output_path, 'seed_clusters', 'final_seed_clusters.png')
            )
            
            logger.info(f"Generated seed cluster visualizations in {os.path.join(output_path, 'seed_clusters')}")
        except Exception as e:
            logger.warning(f"Failed to generate seed cluster visualizations: {str(e)}")
    
    # Final reporting
    logger.info(f"Formed {len(final_seed_clusters)} seed clusters in {time.time() - start_time:.2f} seconds")
    
    return final_seed_clusters

def visualize_seed_clusters(clusters: Dict[str, List[str]], 
                          container_features: Dict[str, Tuple[float, float, int]],
                          feature_matrix: Optional[np.ndarray] = None,
                          cluster_labels: Optional[np.ndarray] = None,
                          container_ids: Optional[List[str]] = None,
                          output_path: str = './seed_clusters.png') -> None:
    """
    Generate a scatter plot visualization of seed clusters.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    feature_matrix : Optional[np.ndarray]
        Original feature matrix used for clustering (if available)
    cluster_labels : Optional[np.ndarray]
        Cluster labels from hierarchical clustering (if available)
    container_ids : Optional[List[str]]
        List of container IDs corresponding to feature_matrix rows (if available)
    output_path : str
        Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    plt.figure(figsize=(12, 8))
    
    # Get a color map with enough distinct colors
    num_clusters = len(clusters)
    cmap = plt.cm.get_cmap('tab20', num_clusters)  # Use tab20 for up to 20 distinct colors
    
    # Determine if we're using the original clustering result or visualizing the clusters dictionary
    if feature_matrix is not None and cluster_labels is not None and container_ids is not None:
        # Method 1: Use the original clustering results
        unique_labels = np.unique(cluster_labels)
        
        # Map cluster indices to nice colors from our colormap
        color_map = {label: cmap(i % num_clusters) for i, label in enumerate(unique_labels)}
        
        # Plot each point with its cluster color
        for i, (container_id, label) in enumerate(zip(container_ids, cluster_labels)):
            if container_id in container_features:
                plt.scatter(
                    feature_matrix[i, 0], 
                    feature_matrix[i, 1],
                    color=color_map[label],
                    s=80,
                    edgecolors='black',
                    alpha=0.8
                )
    else:
        # Method 2: Extract features from clusters dictionary
        for i, (cluster_id, container_ids) in enumerate(clusters.items()):
            # Extract centroids and spans for containers in this cluster
            centroids = []
            spans = []
            
            for container_id in container_ids:
                if container_id in container_features:
                    centroid, span, _ = container_features[container_id]
                    centroids.append(centroid)
                    spans.append(span)
            
            if centroids and spans:
                # Plot each container in this cluster
                plt.scatter(
                    centroids, 
                    spans,
                    color=cmap(i % num_clusters),
                    label=f"Cluster {cluster_id}",
                    s=80,
                    edgecolors='black',
                    alpha=0.8
                )
    
    # Add title and labels
    plt.title('Critical Container Seed Clusters', fontsize=16)
    plt.xlabel('Aisle Centroid', fontsize=14)
    plt.ylabel('Aisle Span', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a legend if we have a reasonable number of clusters
    if num_clusters <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def kmeans_with_seed_centers(non_critical_containers: List[str], 
                             container_features: Dict[str, Tuple[float, float, int]],
                             cluster_centers: Dict[str, Tuple[float, float]],
                             max_iterations: int = 50) -> Dict[str, List[str]]:
    """
    K-means clustering using predefined (fixed) cluster centers from seed clusters.
    This version does not update the centers after the initial assignment.
    
    Parameters
    ----------
    non_critical_containers : List[str]
        List of non-critical container IDs.
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles).
    cluster_centers : Dict[str, Tuple[float, float]]
        Dictionary mapping cluster IDs to fixed centers (centroid, span).
    max_iterations : int, optional
        (Not used in this fixed-center version; maintained for signature compatibility)
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs.
    """
    import time
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info(f"Running fixed-center clustering with {len(cluster_centers)} seed centers on {len(non_critical_containers)} non-critical containers")
    
    # Extract features for non-critical containers
    feature_arrays = []
    valid_containers = []
    
    for c_id in non_critical_containers:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if not valid_containers:
        logger.warning("No valid non-critical containers with features found")
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
        logger.info(f"Fixed-center clustering cluster {cluster_id}: {len(containers)} containers")
    
    logger.info(f"Fixed-center clustering completed in {time.time() - start_time:.2f} seconds")
    return final_clusters

'''def kmeans_with_seed_centers(non_critical_containers: List[str], 
                            container_features: Dict[str, Tuple[float, float, int]],
                            cluster_centers: Dict[str, Tuple[float, float]],
                            max_iterations: int = 50) -> Dict[str, List[str]]:
    """
    K-means clustering using predefined cluster centers from seed clusters.
    
    Parameters
    ----------
    non_critical_containers : List[str]
        List of non-critical container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    cluster_centers : Dict[str, Tuple[float, float]]
        Dictionary mapping cluster IDs to centers (centroid, span)
    max_iterations : int, optional
        Maximum number of iterations for K-means, by default 50
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    start_time = time.time()
    logger.info(f"Running K-means with {len(cluster_centers)} seed centers on {len(non_critical_containers)} non-critical containers")
    
    # Extract features for non-critical containers
    feature_arrays = []
    valid_containers = []
    
    for c_id in non_critical_containers:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if not valid_containers:
        logger.warning("No valid non-critical containers with features found")
        return {k: [] for k in cluster_centers.keys()}
    
    # Convert to numpy array
    feature_matrix = np.array(feature_arrays)
    
    # Prepare cluster ID to index mapping and initial centers
    cluster_id_to_idx = {cid: i for i, cid in enumerate(cluster_centers.keys())}
    idx_to_cluster_id = {i: cid for cid, i in cluster_id_to_idx.items()}
    
    # Set up initial centers from seed clusters
    k = len(cluster_centers)
    initial_centers = np.zeros((k, 2))
    
    for cluster_id, (centroid, span) in cluster_centers.items():
        idx = cluster_id_to_idx[cluster_id]
        initial_centers[idx] = [centroid, span]
    
    # K-means initialization
    assignments = np.zeros(len(valid_containers), dtype=int)
    
    # Initialize assignment to closest center
    for i in range(len(valid_containers)):
        distances = np.linalg.norm(feature_matrix[i] - initial_centers, axis=1)
        closest_cluster = np.argmin(distances)
        assignments[i] = closest_cluster
    
    # K-means iterations
    converged = False
    for iteration in range(max_iterations):
        # Recalculate centers
        new_centers = np.zeros_like(initial_centers)
        cluster_counts = np.zeros(k)
        
        for i in range(len(valid_containers)):
            cluster = assignments[i]
            new_centers[cluster] += feature_matrix[i]
            cluster_counts[cluster] += 1
        
        # Update centers (avoid division by zero)
        for j in range(k):
            if cluster_counts[j] > 0:
                new_centers[j] = new_centers[j] / cluster_counts[j]
            else:
                # Keep original center if no points assigned
                new_centers[j] = initial_centers[j]
        
        # Reassign points
        changes = 0
        for i in range(len(valid_containers)):
            distances = np.linalg.norm(feature_matrix[i] - new_centers, axis=1)
            closest_cluster = np.argmin(distances)
            
            if assignments[i] != closest_cluster:
                assignments[i] = closest_cluster
                changes += 1
        
        # Update centers for next iteration
        initial_centers = new_centers.copy()
        
        # Log progress
        logger.debug(f"K-means iteration {iteration+1}: {changes} changes")
        
        # Check for convergence
        if changes == 0:
            converged = True
            logger.info(f"K-means converged after {iteration+1} iterations")
            break
    
    if not converged:
        logger.info(f"K-means reached maximum iterations ({max_iterations}) without convergence")
    
    # Build final clusters
    final_clusters = {cid: [] for cid in cluster_centers.keys()}
    
    for i, container_id in enumerate(valid_containers):
        cluster_idx = assignments[i]
        cluster_id = idx_to_cluster_id[cluster_idx]
        final_clusters[cluster_id].append(container_id)
    
    # Report cluster sizes
    for cluster_id, containers in final_clusters.items():
        logger.info(f"K-means cluster {cluster_id}: {len(containers)} containers")
    
    logger.info(f"K-means clustering completed in {time.time() - start_time:.2f} seconds")
    return final_clusters'''

def augment_clusters(seed_clusters: Dict[str, List[str]], 
                    non_critical_containers: List[str],
                    critical_containers: List[str],
                    container_features: Dict[str, Tuple[float, float, int]], 
                    max_cluster_size: int,
                    generate_visuals: bool = True,
                    output_path: str = './cluster_analysis') -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Augment seed clusters with non-critical containers up to max_cluster_size.
    
    Parameters
    ----------
    seed_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs (from initial K-means)
    non_critical_containers : List[str]
        List of non-critical container IDs
    critical_containers : List[str]
        List of critical and urgent container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    max_cluster_size : int
        Maximum size for any cluster
    generate_visuals : bool, optional
        Whether to generate visualizations, by default True
    output_path : str, optional
        Path to save visualizations, by default './cluster_analysis'
        
    Returns
    -------
    Tuple[Dict[str, List[str]], List[str]]
        Dictionary mapping cluster IDs to lists of container IDs and list of remaining containers
    """
    start_time = time.time()
    logger.info(f"Augmenting {len(seed_clusters)} clusters with non-critical containers")
    
    # Create visualization output directory if needed
    if generate_visuals:
        augment_viz_path = os.path.join(output_path, 'augmented_clusters')
        os.makedirs(augment_viz_path, exist_ok=True)
        
        # Visualize initial seed clusters before augmentation
        visualize_augmented_clusters(
            seed_clusters, 
            critical_containers, 
            container_features, 
            os.path.join(augment_viz_path, 'before_augmentation.png'),
            title="Seed Clusters Before Augmentation"
        )
    
    # Convert critical containers to a set for faster lookups
    critical_set = set(critical_containers)
    
    # Calculate cluster centers
    cluster_centers = calculate_cluster_centers(seed_clusters, container_features)
    
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
        
        logger.info(f"Cluster {cluster_id}: critical={critical_count}, non-critical={len(non_critical_in_cluster)}, " 
                   f"total={current_total}, space left={space_left}")
        
        # If no space left based on critical containers, remove all non-critical
        if space_left <= 0:
            logger.info(f"Cluster {cluster_id} already exceeds max size with critical containers")
            # Remove all non-critical containers from this cluster
            final_clusters[cluster_id] = critical_in_cluster
            # Add removed containers back to remaining
            for c_id in non_critical_in_cluster:
                remaining.add(c_id)
            continue
        
        # If we already have non-critical containers from K-means, check if we need to remove some
        if len(non_critical_in_cluster) > space_left:
            logger.info(f"Cluster {cluster_id} has too many non-critical containers, removing {len(non_critical_in_cluster) - space_left}")
            
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
            logger.info(f"Cluster {cluster_id} can add {space_left} more non-critical containers")
            
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
            
            logger.info(f"Added {len(containers_to_add)} additional non-critical containers to cluster {cluster_id}")
            
            # Remove from remaining pool
            for c_id in containers_to_add:
                if c_id in remaining:
                    remaining.remove(c_id)
    
    # Log final cluster sizes
    for cluster_id, containers in final_clusters.items():
        critical_count = sum(1 for c_id in containers if c_id in critical_set)
        non_critical_count = len(containers) - critical_count
        
        logger.info(f"Augmented cluster {cluster_id}: {critical_count} critical + {non_critical_count} non-critical = {len(containers)} total")
    
    remaining_list = list(remaining)
    logger.info(f"Augmentation completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Remaining non-critical containers: {len(remaining_list)}")
    
    # Visualize final augmented clusters
    if generate_visuals:
        try:
            visualize_augmented_clusters(
                final_clusters, 
                critical_containers, 
                container_features, 
                os.path.join(augment_viz_path, 'after_augmentation.png'),
                title="Clusters After Augmentation"
            )
            
            # Create a comparison visualization showing cluster changes
            visualize_cluster_comparison(
                seed_clusters,
                final_clusters,
                critical_containers,
                container_features,
                os.path.join(augment_viz_path, 'augmentation_comparison.png')
            )
            
            logger.info(f"Generated augmented cluster visualizations in {augment_viz_path}")
        except Exception as e:
            logger.warning(f"Failed to generate augmented cluster visualizations: {str(e)}")
    
    return final_clusters, remaining_list

def visualize_augmented_clusters(clusters: Dict[str, List[str]],
                               critical_containers: List[str],
                               container_features: Dict[str, Tuple[float, float, int]],
                               output_path: str,
                               title: str = "Augmented Clusters") -> None:
    """
    Visualize the clusters with critical containers marked as crosses.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : List[str]
        List of critical container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    output_path : str
        Path to save the visualization
    title : str, optional
        Title for the plot, by default "Augmented Clusters"
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    try:
        # Create a set of critical containers for faster lookups
        critical_set = set(critical_containers)
        
        plt.figure(figsize=(16, 10))
        
        # Create a colormap with enough colors for all clusters
        num_clusters = len(clusters)
        if num_clusters == 0:
            logger.warning("No clusters to visualize")
            return
            
        # Use a qualitative colormap for discrete colors
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
                    linewidths=0.5,
                    label=f"Cluster {cluster_id} (non-critical)"
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
                    linewidths=1.0,
                    label=f"Cluster {cluster_id} (critical)"
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
                (avg_centroid, avg_span),
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Add markers to legend for critical and non-critical
        from matplotlib.lines import Line2D
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
        plt.title(title, fontsize=14)
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
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate statistics visualization
        visualize_cluster_stats(clusters, critical_containers, container_features, 
                              os.path.dirname(output_path))
        
    except Exception as e:
        logger.error(f"Error generating augmented clusters visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_cluster_stats(clusters: Dict[str, List[str]],
                          critical_containers: List[str],
                          container_features: Dict[str, Tuple[float, float, int]],
                          output_path: str) -> None:
    """
    Generate a visualization of cluster statistics.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : List[str]
        List of critical container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    output_path : str
        Directory to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
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
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'total': total_containers,
                'critical': critical_count,
                'non_critical': non_critical_count,
                'critical_pct': critical_pct,
                'avg_centroid': avg_centroid,
                'avg_span': avg_span
            })
        
        # Sort clusters by ID
        cluster_stats.sort(key=lambda x: x['cluster_id'])
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # Bar colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_stats)))
        
        # Plot 1: Container counts (stacked bars)
        x = range(len(cluster_stats))
        cluster_ids = [stats['cluster_id'] for stats in cluster_stats]
        critical_counts = [stats['critical'] for stats in cluster_stats]
        non_critical_counts = [stats['non_critical'] for stats in cluster_stats]
        
        # Plot stacked bars
        ax1.bar(x, critical_counts, label='Critical', color='red', alpha=0.7)
        ax1.bar(x, non_critical_counts, bottom=critical_counts, label='Non-Critical', color='blue', alpha=0.7)
        
        # Add labels
        for i, stats in enumerate(cluster_stats):
            # Total label at top
            ax1.text(i, stats['total'] + 1, str(stats['total']), 
                   ha='center', va='bottom', fontsize=9)
            
            # Critical label in middle of its section
            if stats['critical'] > 0:
                ax1.text(i, stats['critical'] / 2, str(stats['critical']), 
                       ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            
            # Non-critical label in middle of its section
            if stats['non_critical'] > 0:
                ax1.text(i, stats['critical'] + stats['non_critical'] / 2, str(stats['non_critical']), 
                       ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        
        ax1.set_title('Container Counts by Cluster', fontsize=14)
        ax1.set_xlabel('Cluster ID', fontsize=12)
        ax1.set_ylabel('Number of Containers', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cluster_ids)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Critical percentage and average metrics
        cluster_ids = [stats['cluster_id'] for stats in cluster_stats]
        critical_pcts = [stats['critical_pct'] for stats in cluster_stats]
        
        # Create a second axis for the centroid and span
        ax3 = ax2.twinx()
        
        # Plot critical percentage as bars
        bars = ax2.bar(x, critical_pcts, color=colors, alpha=0.7)
        
        # Plot average centroid and span as lines
        avg_centroids = [stats['avg_centroid'] for stats in cluster_stats]
        avg_spans = [stats['avg_span'] for stats in cluster_stats]
        
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
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'cluster_statistics.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating cluster statistics visualization: {str(e)}")

def visualize_cluster_comparison(before_clusters: Dict[str, List[str]],
                               after_clusters: Dict[str, List[str]],
                               critical_containers: List[str],
                               container_features: Dict[str, Tuple[float, float, int]],
                               output_path: str) -> None:
    """
    Create a visualization comparing clusters before and after augmentation.
    
    Parameters
    ----------
    before_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs before augmentation
    after_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs after augmentation
    critical_containers : List[str]
        List of critical container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    output_path : str
        Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np
    
    try:
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
                    (avg_centroid, avg_span),
                    fontsize=10,
                    weight='bold',
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
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating cluster comparison visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

'''def calculate_tours(clusters: Dict[str, List[str]], 
                   critical_containers: List[str], 
                   containers_per_tour: int = 20) -> Dict[str, int]:
    """
    Calculate tours for each cluster based on critical containers.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : List[str]
        List of critical container IDs
    containers_per_tour : int, optional
        Maximum containers per tour, by default 20
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping cluster IDs to number of tours
    """
    critical_set = set(critical_containers)
    cluster_tours = {}
    total_tours = 0
    
    for cluster_id, container_ids in clusters.items():
        # Count critical containers in this cluster
        critical_count = sum(1 for c_id in container_ids if c_id in critical_set)
        
        # Calculate tours (at least 1, rounded up for critical containers)
        tours = max(1, (critical_count + containers_per_tour - 1) // containers_per_tour)
        
        # Store number of tours
        cluster_tours[cluster_id] = tours
        total_tours += tours
        
        logger.debug(f"Cluster {cluster_id}: {critical_count} critical containers, {tours} tours")
    
    logger.info(f"Total tours across all clusters: {total_tours}")
    return cluster_tours'''

def calculate_cluster_quality(cluster_containers: List[str], 
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
    
    if len(valid_containers) < 20:
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
    span_factor = 0.5  # Less weight on span
    
    # Inverse of weighted sum (smaller values = higher quality)
    weighted_sum = (centroid_factor * (centroid_std + avg_centroid_dist) + 
                   span_factor * (span_std + avg_span_dist))
    
    # Convert to quality score (higher is better)
    # Add 0.1 to avoid division by zero
    quality = 1.0 / (weighted_sum + 0.1)
    
    return quality

def form_additional_clusters(remaining_containers: List[str], 
                           container_data: pd.DataFrame,
                           container_features: Dict[str, Tuple[float, float, int]], 
                           max_cluster_size: int) -> Dict[str, List[str]]:
    """
    Form additional clusters from remaining containers using hierarchical clustering.
    Iteratively breaks down clusters until all clusters are below max_cluster_size.
    
    Parameters
    ----------
    remaining_containers : List[str]
        List of remaining container IDs
    container_data : pd.DataFrame
        Container data with SKU information
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    max_cluster_size : int
        Maximum size for any cluster
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    start_time = time.time()
    logger.info(f"Forming additional clusters from {len(remaining_containers)} remaining containers")
    
    # Skip if no remaining containers
    if not remaining_containers:
        logger.info("No remaining containers to cluster")
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
        logger.warning("No valid remaining containers with features")
        return {}
    
    # Convert to numpy array
    feature_matrix = np.array(feature_arrays)
    
    # Normalize features
    normalized_features = normalize_features(feature_matrix)
    
    # Estimate optimal number of clusters based on max_cluster_size
    estimated_clusters = max(2, len(valid_containers) // max_cluster_size + 1)
    max_possible_clusters = min(10, len(valid_containers) // 2)
    
    # Determine optimal number of clusters
    optimal_clusters, _ = determine_optimal_clusters(
        normalized_features, 
        min_clusters=2, 
        max_clusters=min(max_possible_clusters, estimated_clusters),
        max_cluster_size = max_cluster_size   
    )
    
    logger.info(f"Determined optimal number of additional clusters: {optimal_clusters}")
    
    # Perform hierarchical clustering
    Z = linkage(normalized_features, method='ward')
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
        logger.info(f"Iteration {iteration}: Processing {len(clusters_to_process)} clusters")
        
        next_clusters_to_process = {}
        
        for cluster_id, containers in clusters_to_process.items():
            # If the cluster is small enough, add it to the final results
            if len(containers) <= max_cluster_size:
                final_clusters[cluster_id] = containers
                logger.debug(f"Cluster {cluster_id} size {len(containers)} is within limit")
                continue
                
            # Otherwise, subdivide this cluster
            logger.info(f"Subdividing cluster {cluster_id} with {len(containers)} containers")
            
            # Build feature matrix for this specific cluster
            sub_feature_matrix = []
            sub_containers = []
            
            for c_id in containers:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    sub_feature_matrix.append([centroid, span])
                    sub_containers.append(c_id)
            
            if not sub_containers:
                logger.warning(f"No valid containers in cluster {cluster_id}")
                continue
                
            # Convert to numpy and normalize
            sub_features = normalize_features(np.array(sub_feature_matrix))
            
            # Calculate optimal sub-clusters - aim for smaller clusters this time
            sub_max_clusters = max(2, len(sub_containers) // (max_cluster_size // 2) + 1)
            sub_optimal_clusters, _ = determine_optimal_clusters(
                sub_features, min_clusters=2, max_clusters=sub_max_clusters,max_cluster_size = max_cluster_size   
            )
            
            # Apply clustering
            sub_Z = linkage(sub_features, method='ward')
            sub_labels = fcluster(sub_Z, sub_optimal_clusters, criterion='maxclust')
            
            # Create sub-clusters
            sub_cluster_groups = {}
            for i, container_id in enumerate(sub_containers):
                sub_cluster_id = int(sub_labels[i])
                if sub_cluster_id not in sub_cluster_groups:
                    sub_cluster_groups[sub_cluster_id] = []
                sub_cluster_groups[sub_cluster_id].append(container_id)
            
            # Add sub-clusters to appropriate collections
            for sub_id, sub_containers in sub_cluster_groups.items():
                sub_key = f"{cluster_id}_{sub_id}"
                
                # If sub-cluster is small enough, add to final results
                if len(sub_containers) <= max_cluster_size:
                    final_clusters[sub_key] = sub_containers
                    logger.debug(f"Sub-cluster {sub_key} size {len(sub_containers)} is within limit")
                else:
                    # Otherwise, add to the next iteration for further subdivision
                    next_clusters_to_process[sub_key] = sub_containers
                    logger.debug(f"Sub-cluster {sub_key} size {len(sub_containers)} needs further subdivision")
        
        # Check if we have clusters that still need processing
        if not next_clusters_to_process:
            logger.info(f"All clusters are within size limit after iteration {iteration}")
            break
            
        clusters_to_process = next_clusters_to_process
    
    # Handle any remaining large clusters if we hit the iteration limit
    if clusters_to_process:
        logger.warning(f"Reached max iterations ({max_iterations}), {len(clusters_to_process)} clusters still exceed size limit")
        # Add remaining clusters to the final result regardless of size
        for cluster_id, containers in clusters_to_process.items():
            final_clusters[f"{cluster_id}_forced"] = containers
    
    # Log final statistics
    cluster_sizes = [len(containers) for containers in final_clusters.values()]
    if cluster_sizes:
        logger.info(f"Final clusters: {len(final_clusters)}")
        logger.info(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
        logger.info(f"Max cluster size: {max(cluster_sizes)}")
        logger.info(f"Min cluster size: {min(cluster_sizes)}")
        
        # Check if all clusters are within limit
        oversized = [size for size in cluster_sizes if size > max_cluster_size]
        if oversized:
            logger.warning(f"{len(oversized)} clusters still exceed max_cluster_size")
        else:
            logger.info("All clusters are within max_cluster_size limit")
    
    logger.info(f"Formed {len(final_clusters)} final clusters in {time.time() - start_time:.2f} seconds")
    return final_clusters


def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize features by column to [0,1] range
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix of features to normalize
        
    Returns
    -------
    np.ndarray
        Normalized feature matrix
    """
    # Skip normalization if empty
    if feature_matrix.size == 0:
        return feature_matrix
        
    # Create a copy to avoid modifying the original
    normalized = feature_matrix.copy()
    
    # Normalize each column
    for col in range(feature_matrix.shape[1]):
        col_min = np.min(feature_matrix[:, col])
        col_max = np.max(feature_matrix[:, col])
        
        # Avoid division by zero
        if col_max > col_min:
            normalized[:, col] = (feature_matrix[:, col] - col_min) / (col_max - col_min)
    
    return normalized

def select_additional_clusters(additional_clusters: Dict[str, List[str]], 
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
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
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
    logger.info(f"Selecting additional clusters to fill remaining capacity of {remaining_capacity} tours")
    
    # Skip if no capacity or no clusters
    if remaining_capacity <= 0 or not additional_clusters:
        logger.info("No remaining capacity or no additional clusters to select")
        return {}
    
    # Calculate metrics for each cluster
    cluster_metrics = {}
    
    for cluster_id, containers in additional_clusters.items():
        quality = calculate_cluster_quality(containers, container_features)     #bookmark - replace to use some cluster stats
        
        # Calculate tours based on total containers
        tours = math.floor(len(containers) / containers_per_tour)
        
        cluster_metrics[cluster_id] = {
            'containers': containers,
            'quality': quality,
            'tours': tours,
            'size': len(containers)
        }
        
        logger.debug(f"Cluster {cluster_id}: {len(containers)} containers, {tours} tours, quality={quality:.4f}")
    
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
            
            logger.debug(f"Selected cluster {cluster_id}: +{info['tours']} tours, total now {total_tours}/{remaining_capacity}")
        
        # Break if we've reached capacity
        if total_tours >= remaining_capacity:
            break
    
    logger.info(f"Selected {len(selected)} additional clusters with {total_tours} tours and {total_containers} containers")
    logger.info(f"Selection completed in {time.time() - start_time:.2f} seconds")
    
    return selected

def merge_cluster_results(seed_clusters: Dict[str, Dict[str, Any]], 
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
    logger.info(f"Merging {len(seed_clusters)} seed clusters with {len(additional_clusters)} additional clusters")
    
    # Start with seed clusters
    final_clusters = seed_clusters.copy()
    
    # Add additional clusters with new IDs to avoid collisions
    if additional_clusters:
        # Find maximum existing ID
        max_id = -1
        for cluster_id in seed_clusters.keys():
            try:
                # Handle numeric IDs and IDs with underscores
                if '_' in cluster_id:
                    id_part = cluster_id.split('_')[0]
                    if id_part.isdigit():
                        max_id = max(max_id, int(id_part))
                elif cluster_id.isdigit():
                    max_id = max(max_id, int(cluster_id))
            except (ValueError, IndexError):
                continue
        
        # Add additional clusters with new IDs
        next_id = max_id + 1
        for info in additional_clusters.values():
            final_clusters[str(next_id)] = info
            next_id += 1
    
    # Count total containers and tours
    total_containers = sum(len(info['containers']) for info in final_clusters.values())
    total_tours = sum(info['tours'] for info in final_clusters.values())
    
    logger.info(f"Final result: {len(final_clusters)} clusters with {total_containers} containers and {total_tours} tours")
    logger.info(f"Merge completed in {time.time() - start_time:.2f} seconds")
    
    return final_clusters

def enhanced_container_clustering(container_data: pd.DataFrame, 
                                slotbook_data: pd.DataFrame,
                                max_cluster_size: int, 
                                min_clusters: int, 
                                max_clusters: int,
                                containers_per_tour: int = 20,
                                generate_visuals: bool = True,
                                output_path: str = './cluster_analysis',
                                picking_capacity: int = 1000) -> Dict[str, List[str]]:
    """
    Enhanced clustering algorithm that prioritizes critical containers.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    max_cluster_size : int
        Maximum size for any cluster
    min_clusters : int
        Minimum number of clusters
    max_clusters : int
        Maximum number of clusters
    containers_per_tour : int, optional
        Maximum containers per tour, by default 20
    generate_visuals : bool, optional
        Whether to generate visualizations, by default True
    output_path : str, optional
        Path to save visualizations, by default './cluster_analysis'
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    start_time = time.time()
    logger.info("Starting enhanced clustering with critical container prioritization")
    
    # Step 0: Prepare data structures and get containers
    container_ids = container_data['container_id'].unique().tolist()
    logger.info(f"Total containers to process: {len(container_ids)}")
    
    # Build SKU-aisle mapping
    sku_aisle_mapping = {}
    for _, row in slotbook_data.iterrows():
        if row['item_number'] not in sku_aisle_mapping:
            sku_aisle_mapping[row['item_number']] = []
        sku_aisle_mapping[row['item_number']].append(row['aisle_sequence'])
    
    # Extract container features
    container_features = extract_container_features(container_data, sku_aisle_mapping, container_ids)
    
    # Step 0: Check for critical containers
    critical_containers = get_critical_containers(container_data, container_ids)
    
    # === MODIFIED BRANCHING LOGIC ===
    # If no critical containers OR if all containers fit within capacity, 
    # skip directly to step 7 to form clusters from all containers
    if not critical_containers or len(container_ids) <= picking_capacity:
        logger.info(f"{'No critical containers' if not critical_containers else 'All containers fit within capacity'} - forming direct clusters")
        
        # Skip to step 7: Form clusters from all containers
        all_container_clusters = form_additional_clusters(
            container_ids, container_data, container_features, max_cluster_size
        )
        
        # Calculate how many tours we can allocate
        max_tours = min(max_clusters, picking_capacity // containers_per_tour)
        
        # Select best clusters to fill available capacity
        selected_clusters = select_additional_clusters(
            all_container_clusters, container_features, max_tours, containers_per_tour
        )
        
        # Extract just the container lists for the return value
        final_clusters = {cluster_id: info['containers'] for cluster_id, info in selected_clusters.items()}
        
        # Renumber clusters sequentially and display statistics
        return finalize_clusters(
            final_clusters, 
            critical_containers,  # This may be empty but we still pass it for consistency
            container_features,
            containers_per_tour,
            generate_visuals,
            output_path
        )
    
    # Continue with original critical container prioritization logic if:
    # 1. There are critical containers AND
    # 2. Total containers exceed picking capacity
    
    # Adjust min_clusters if needed for critical containers
    min_clusters = ((len(critical_containers) // max_cluster_size) + 
                     (1 if len(critical_containers) % max_cluster_size > 0 else 0))
    
    logger.info(f"Adjusted minimum clusters to {min_clusters} based on critical containers")
    
    # Step 1: Separate containers
    non_critical_containers = [c_id for c_id in container_ids if c_id not in critical_containers]
    logger.info(f"Separated {len(critical_containers)} critical and {len(non_critical_containers)} non-critical containers")
    
    # If critical containers exceed picking capacity, prioritize them
    if len(critical_containers) > picking_capacity:
        logger.warning(f"Critical containers ({len(critical_containers)}) exceed picking capacity ({picking_capacity})")
        # We'll just use all critical containers
    
    # Step 2: Form seed clusters from critical containers
    seed_clusters = form_seed_clusters(
        critical_containers, 
        container_data, 
        container_features, 
        max_cluster_size,
        generate_visuals=generate_visuals,  
        output_path=output_path             
    )
    
    if not seed_clusters:
        logger.error("Failed to form seed clusters from critical containers")
        raise ValueError("No seed clusters formed from critical containers")
    
    # Step 3: Calculate cluster centers for the seed clusters
    cluster_centers = calculate_cluster_centers(seed_clusters, container_features)
    
    # Step 4: Run K-means with seed centers on non-critical containers
    kmeans_clusters = kmeans_with_seed_centers(non_critical_containers, container_features, cluster_centers)
    
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
        
        logger.debug(f"Merged cluster {cluster_id}: {len(critical_in_cluster)} critical + {len(non_critical_in_cluster)} non-critical")
    
    # Remove non-critical containers already assigned
    remaining_non_critical = list(set(non_critical_containers) - 
                                set().union(*[set(c) for c in kmeans_clusters.values()]))
    
    # Augment merged clusters with best remaining non-critical containers
    augmented_clusters, remaining_containers = augment_clusters(
        merged_clusters, 
        remaining_non_critical,
        critical_containers,  # Pass the critical containers list
        container_features, 
        max_cluster_size,
        generate_visuals=generate_visuals,  # Pass the visualization flag
        output_path=output_path  # Pass the output path
    )
    
    # Step 6: Calculate tours for seed clusters
    cluster_tours = calculate_tours(augmented_clusters, critical_containers, containers_per_tour)
    
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
    
    if total_seed_tours >= max_clusters:
        logger.info(f"Seed clusters already use all available capacity ({total_seed_tours} >= {max_clusters})")
        # Extract container lists before returning
        seed_clusters_only = {cluster_id: info['containers'] for cluster_id, info in seed_result.items()}
        
        # Before returning, renumber the clusters sequentially and display statistics
        return finalize_clusters(
            seed_clusters_only, 
            critical_containers, 
            container_features,
            containers_per_tour,
            generate_visuals,
            output_path
        )
    
    # Step 7: Handle remaining capacity with additional clusters
    remaining_capacity = (picking_capacity//containers_per_tour) - (total_seed_tours)               
    if remaining_capacity > 0 and remaining_containers:
        # Form additional clusters from remaining containers
        additional_clusters = form_additional_clusters(
            remaining_containers, container_data, container_features, max_cluster_size
        )
        
        # Select best additional clusters to fill remaining capacity
        selected_additional = select_additional_clusters(                  
            additional_clusters, container_features, remaining_capacity, containers_per_tour
        )
        
        # Step 8: Merge seed clusters with selected additional clusters
        final_result = merge_cluster_results(seed_result, selected_additional)
    else:
        logger.info("No remaining capacity or no remaining containers")
        final_result = seed_result
    
    # Extract just the container lists for the return value
    final_clusters = {cluster_id: info['containers'] for cluster_id, info in final_result.items()}
    
    # Before returning, renumber the clusters sequentially and display statistics
    return finalize_clusters(
        final_clusters, 
        critical_containers, 
        container_features,
        containers_per_tour,
        generate_visuals,
        output_path
    )

def finalize_clusters(clusters: Dict[str, List[str]], 
                    critical_containers: List[str],
                    container_features: Dict[str, Tuple[float, float, int]],
                    containers_per_tour: int = 20,
                    generate_visuals: bool = False,
                    output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Renumber clusters sequentially, display detailed statistics for each cluster,
    and generate visualizations if requested.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Original clusters with arbitrary IDs
    critical_containers : List[str]
        List of critical container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    containers_per_tour : int, optional
        Maximum containers per tour, by default 20
    generate_visuals : bool, optional
        Whether to generate visualizations, by default False
    output_path : str, optional
        Path to save visualizations, by default './cluster_analysis'
    
    Returns
    -------
    Dict[str, List[str]]
        Renumbered clusters with sequential IDs (1 through n)
    """
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
    
    # Display statistics for each cluster
    logger.info("\nCluster Statistics:")
    for stats in cluster_stats:
        logger.info(f"Cluster {stats['ClusterID']}: {stats['TotalContainers']} containers "
                   f"({stats['CriticalContainers']} critical, {stats['NonCriticalContainers']} non-critical, "
                   f"{stats['CriticalPercentage']:.1f}% critical) - {stats['NumTours']} tours")
        logger.info(f"  Aisle range: {stats['MinCentroid']:.1f} - {stats['MaxCentroid']:.1f}, "
                   f"avg centroid: {stats['AvgCentroid']:.1f}, avg span: {stats['AvgSpan']:.1f}")
    
    # Overall statistics
    total_containers = sum(len(containers) for containers in renumbered_clusters.values())
    total_critical = sum(sum(1 for c_id in containers if c_id in critical_set) 
                        for containers in renumbered_clusters.values())
    total_tours = sum(stats['NumTours'] for stats in cluster_stats)
    
    logger.info("\nOverall Statistics:")
    logger.info(f"Total clusters: {len(renumbered_clusters)}")
    logger.info(f"Total containers: {total_containers}")
    logger.info(f"Critical containers: {total_critical} ({100 * total_critical / total_containers:.1f}%)")
    logger.info(f"Non-critical containers: {total_containers - total_critical} ({100 * (total_containers - total_critical) / total_containers:.1f}%)")
    logger.info(f"Total tours required: {total_tours}")
    
    # Create a pretty-printed tabular display
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
        
        # Add a totals row
        total_containers = sum(len(containers) for containers in renumbered_clusters.values())
        total_critical = sum(sum(1 for c_id in containers if c_id in critical_set) 
                            for containers in renumbered_clusters.values())
        total_non_critical = total_containers - total_critical
        critical_pct = 100 * total_critical / total_containers if total_containers else 0
        
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
        logger.info("\nDetailed Cluster Statistics Table:")
        logger.info("\n" + table)
        
    except (ImportError, Exception) as e:
        logger.debug(f"Could not create tabular statistics: {str(e)}")
        # Fallback to simpler display if tabulate isn't available
        logger.info("\nCluster Statistics Summary:")
        for stats in cluster_stats:
            logger.info(f"Cluster {stats['ClusterID']}: {stats['TotalContainers']} containers "
                       f"({stats['CriticalContainers']} critical, {stats['NonCriticalContainers']} non-critical) - {stats['NumTours']} tours")
    
    # Generate visualizations if requested
    if generate_visuals:
        try:
            # Call visualization function from container_clustering.py
            generate_final_cluster_visualization(
                renumbered_clusters, 
                container_features, 
                output_path,
                use_distinct_aisles=False,  # Using aisle span as secondary feature
                critical_containers=critical_containers  # Pass critical containers list
            )
            logger.info(f"Generated final cluster visualization in {output_path}")
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {str(e)}")
    
    return renumbered_clusters, cluster_stats_df


def calculate_tours(clusters: Dict[str, List[str]], 
                   critical_containers: List[str], 
                   containers_per_tour: int = 20) -> Dict[str, int]:
    """
    Calculate tours for each cluster based on critical containers.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : List[str]
        List of critical container IDs
    containers_per_tour : int, optional
        Maximum containers per tour, by default 20
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping cluster IDs to number of tours
    """
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
            # For clusters without critical containers, simply divide total count
            tours = max(1, math.floor(len(container_ids) / containers_per_tour))
        
        # Store number of tours
        cluster_tours[cluster_id] = tours
        total_tours += tours
        
        logger.debug(f"Cluster {cluster_id}: {critical_count} critical containers, {tours} tours")
    
    logger.info(f"Total tours across all clusters: {total_tours}")
    return cluster_tours

def standard_hierarchical_clustering(container_data: pd.DataFrame, 
                                   slotbook_data: pd.DataFrame,
                                   container_ids: List[str],
                                   max_cluster_size: int,
                                   min_clusters: int,
                                   max_clusters: int,
                                   generate_visuals: bool = False,
                                   output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Standard hierarchical clustering implementation (fallback when no critical containers).
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    container_ids : List[str]
        List of container IDs to cluster
    max_cluster_size : int
        Maximum size for any cluster
    min_clusters : int
        Minimum number of clusters
    max_clusters : int
        Maximum number of clusters
    generate_visuals : bool, optional
        Whether to generate visualizations, by default False
    output_path : str, optional
        Path to save visualizations, by default './cluster_analysis'
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    logger.info(f"Running standard hierarchical clustering on {len(container_ids)} containers")
    
    # Build SKU-aisle mapping
    sku_aisle_mapping = {}
    for _, row in slotbook_data.iterrows():
        if row['item_number'] not in sku_aisle_mapping:
            sku_aisle_mapping[row['item_number']] = []
        sku_aisle_mapping[row['item_number']].append(row['aisle_sequence'])
    
    # Extract features
    container_features = extract_container_features(container_data, sku_aisle_mapping, container_ids)
    
    # Get feature matrix
    feature_arrays = []
    valid_containers = []
    
    for c_id in container_ids:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    if not valid_containers:
        logger.warning("No valid containers with features")
        return {}
    
    # Convert to numpy array
    feature_matrix = np.array(feature_arrays)
    
    # Normalize features
    normalized_features = normalize_features(feature_matrix)
    
    # Determine optimal number of clusters
    optimal_clusters, _ = determine_optimal_clusters(
        normalized_features, min_clusters, max_clusters, max_cluster_size
    )
    
    logger.info(f"Determined optimal number of clusters: {optimal_clusters}")
    
    # Perform hierarchical clustering
    Z = linkage(normalized_features, method='ward')
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to clusters
    clusters = {}
    for i, container_id in enumerate(valid_containers):
        cluster_id = int(cluster_labels[i])
        if str(cluster_id) not in clusters:
            clusters[str(cluster_id)] = []
        clusters[str(cluster_id)].append(container_id)
    
    # Apply sub-clustering for any clusters exceeding max_cluster_size
    final_clusters = {}
    
    for cluster_id, containers in clusters.items():
        if len(containers) > max_cluster_size:
            # Apply hierarchical sub-clustering
            sub_clusters = hierarchical_sub_clustering(
                container_data, None, containers, None,
                cluster_id, max_cluster_size, False, 2, 5, 
                sku_aisle_mapping=sku_aisle_mapping
            )
            final_clusters.update(sub_clusters)
        else:
            final_clusters[cluster_id] = containers
    
    # Generate visualizations if requested
    if generate_visuals:
        try:
            # Call visualization functions from container_clustering.py
            generate_final_cluster_visualization(
            final_clusters, 
            container_features, 
            output_path,
            use_distinct_aisles=False,  # Using aisle span as secondary feature
            critical_containers=None  # Pass critical containers list
        )
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {str(e)}")
    
    return final_clusters

def hierarchical_sub_clustering(container_data: pd.DataFrame, 
                              slotbook_data: Optional[pd.DataFrame],
                              container_ids: List[str], 
                              container_features: Optional[Dict[str, Tuple[float, float, int]]],
                              parent_cluster_id: str, 
                              max_cluster_size: int, 
                              use_distinct_aisles: bool, 
                              min_clusters: int, 
                              max_clusters: int,
                              depth: int = 0,
                              sku_aisle_mapping: Optional[Dict[str, List[int]]] = None) -> Dict[str, List[str]]:
    """
    Apply sub-clustering to a large cluster to divide it into smaller clusters.
    
    This is a modified version of the sub-clustering function from the original code,
    adapted to work with our enhanced clustering algorithm.
    
    Parameters
    ----------
    [parameters from original function]
    sku_aisle_mapping : Optional[Dict[str, List[int]]]
        Mapping of SKUs to aisles (provided for efficiency)
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    # Base case: if cluster size is within limit or we have too few containers to subdivide
    if len(container_ids) <= max_cluster_size or len(container_ids) <= min_clusters:
        return {f"{parent_cluster_id}": container_ids}
    
    indent = "  " * depth
    logger.info(f"{indent}Sub-clustering {len(container_ids)} containers in cluster {parent_cluster_id} (depth {depth})")
    
    # If container_features wasn't provided, extract them
    if container_features is None:
        # If sku_aisle_mapping wasn't provided, build it
        if sku_aisle_mapping is None and slotbook_data is not None:
            sku_aisle_mapping = {}
            for _, row in slotbook_data.iterrows():
                if row['item_number'] not in sku_aisle_mapping:
                    sku_aisle_mapping[row['item_number']] = []
                sku_aisle_mapping[row['item_number']].append(row['aisle_sequence'])
        elif sku_aisle_mapping is None:
            logger.error("Neither container_features nor slotbook_data/sku_aisle_mapping provided")
            raise ValueError("Cannot perform sub-clustering without feature data")
            
        container_features = extract_container_features(container_data, sku_aisle_mapping, container_ids)
    
    # Get feature matrix
    feature_arrays = []
    valid_containers = []
    
    for c_id in container_ids:
        if c_id in container_features:
            centroid, span, _ = container_features[c_id]
            feature_arrays.append([centroid, span])
            valid_containers.append(c_id)
    
    # If no valid containers, return as is
    if not valid_containers:
        return {f"{parent_cluster_id}": container_ids}
    
    # Convert to numpy array
    feature_matrix = np.array(feature_arrays)
    
    # Normalize features
    normalized_features = normalize_features(feature_matrix)
    
    # Determine optimal number of sub-clusters
    estimated_clusters = max(min_clusters, len(valid_containers) // max_cluster_size + 1)
    max_possible_clusters = min(max_clusters, len(valid_containers) // 2)
    
    optimal_clusters, _ = determine_optimal_clusters(
        normalized_features, 
        min_clusters=min_clusters,
        max_clusters=min(max_possible_clusters, estimated_clusters),
        max_cluster_size = max_cluster_size               
    )
    
    logger.info(f"{indent}Determined optimal number of sub-clusters: {optimal_clusters}")
    
    # Perform hierarchical clustering
    Z = linkage(normalized_features, method='ward')
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to sub-clusters
    sub_clusters = {}
    for i, container_id in enumerate(valid_containers):
        cluster_id = int(cluster_labels[i])
        sub_key = f"{parent_cluster_id}_{cluster_id}"
        
        if sub_key not in sub_clusters:
            sub_clusters[sub_key] = []
            
        sub_clusters[sub_key].append(container_id)
    
    # Check each sub-cluster for size and recurse if needed
    final_clusters = {}
    
    for sub_id, containers in sub_clusters.items():
        # Log sub-cluster statistics
        logger.info(f"{indent}Created sub-cluster {sub_id} with {len(containers)} containers")
        
        # Recursively apply clustering if still too large
        if len(containers) > max_cluster_size:
            logger.info(f"{indent}Sub-cluster {sub_id} exceeds max size, further sub-clustering...")
            next_level_clusters = hierarchical_sub_clustering(
                container_data, slotbook_data, containers, container_features,
                sub_id, max_cluster_size, use_distinct_aisles,
                min_clusters, max_clusters, depth + 1,
                sku_aisle_mapping=sku_aisle_mapping
            )
            final_clusters.update(next_level_clusters)
        else:
            final_clusters[sub_id] = containers
    
    return final_clusters


def modified_cluster_containers(container_data: pd.DataFrame, 
                            slotbook_data: pd.DataFrame, 
                            containers_per_tour: int = 20,
                            max_cluster_size: int = 500,
                            use_distinct_aisles: bool = True, 
                            generate_visuals: bool = True, 
                            output_path: str = './cluster_analysis',
                            prioritize_critical: bool = True,
                            max_picking_capacity: int = 1000) -> Dict[str, List[str]]:
    """
    Modified entry point for container clustering with option to prioritize critical containers.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    containers_per_tour : int
        Maximum containers per tour
    max_cluster_size : int
        Maximum size for any cluster
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature
    min_clusters : int
        Minimum number of clusters to consider
    max_clusters : int
        Maximum number of clusters to consider
    generate_visuals : bool
        Whether to generate visualizations
    output_path : str
        Path to save visualizations
    prioritize_critical : bool
        Whether to use the enhanced algorithm that prioritizes critical containers
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    min_clusters = math.ceil(max_picking_capacity / max_cluster_size)
    max_clusters = int(max_picking_capacity / containers_per_tour)
    # Create output directory if visualizations are enabled
    if generate_visuals:
        os.makedirs(output_path, exist_ok=True)
    
    # Build SKU-to-aisle mapping
    sku_aisle_mapping = {}
    for _, row in slotbook_data.iterrows():
        if row['item_number'] not in sku_aisle_mapping:
            sku_aisle_mapping[row['item_number']] = []
        sku_aisle_mapping[row['item_number']].append(row['aisle_sequence'])
        sku_aisle_mapping[row['item_number']].sort()
    
    # Decide which clustering method to use
    if prioritize_critical:
        logger.info("Using enhanced clustering algorithm with critical container prioritization")
        return enhanced_container_clustering(
            container_data=container_data,
            slotbook_data=slotbook_data,
            max_cluster_size=max_cluster_size,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            containers_per_tour=containers_per_tour,
            generate_visuals=generate_visuals,
            output_path=output_path,
            picking_capacity = max_picking_capacity
        )
    else:
        logger.info("Using original clustering algorithm")
        # Call the original cluster_containers implementation
        return original_cluster_containers(
            container_data=container_data,
            slotbook_data=slotbook_data,
            containers_per_tour=containers_per_tour,
            max_cluster_size=max_cluster_size,
            use_distinct_aisles=use_distinct_aisles,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            generate_visuals=generate_visuals,
            output_path=output_path
        )

def original_cluster_containers(container_data: pd.DataFrame, 
                              slotbook_data: pd.DataFrame,
                              batch_size: Optional[int] = None, 
                              containers_per_tour: int = 20,
                              max_cluster_size: int = 500,
                              use_distinct_aisles: bool = True, 
                              min_clusters: int = 2, 
                              max_clusters: int = 10,
                              generate_visuals: bool = False, 
                              output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Wrapper for the original cluster_containers function to maintain compatibility.
    
    This function calls the existing implementation from container_clustering.py.
    
    Parameters
    ----------
    [parameters same as cluster_containers]
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    # This function would call the existing implementation
    # For the actual implementation, this would call the original function
    from tour_formation.container_clustering_old import cluster_containers as original_implementation
    
    return original_implementation(
        container_data=container_data,
        slotbook_data=slotbook_data,
        containers_per_tour=containers_per_tour,
        max_cluster_size=max_cluster_size,
        use_distinct_aisles=use_distinct_aisles,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        generate_visuals=generate_visuals,
        output_path=output_path
    )

def determine_optimal_clusters(feature_matrix: np.ndarray, 
                             min_clusters: int = 2, 
                             max_clusters: int = 10,
                             max_cluster_size: int = 200) -> Tuple[int, Dict[int, float]]:
    """
    Determine optimal number of clusters using silhouette analysis.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix for containers
    min_clusters : int
        Minimum number of clusters to consider
    max_clusters : int
        Maximum number of clusters to consider
    max_sample_size : int
        Maximum sample size to use for silhouette analysis (for performance)
        
    Returns
    -------
    Tuple[int, Dict[int, float]]
        Optimal number of clusters and dictionary of silhouette scores by cluster count
    """
    from sklearn.metrics import silhouette_score
    
    '''# If we have too many containers, sample to improve performance
    if feature_matrix.shape[0] > max_sample_size:
        logger.info(f"Sampling {max_sample_size} containers for silhouette analysis")
        indices = np.random.choice(feature_matrix.shape[0], max_sample_size, replace=False)
        sample_features = feature_matrix[indices]
    else:'''
    sample_features = feature_matrix
    
    # Adjust max_clusters based on data size
    max_clusters = min(max_clusters, sample_features.shape[0] // max_cluster_size)
    max_clusters = int(max(min_clusters + 1, max_clusters))  # Ensure at least 2 values to compare
    
    logger.debug(f"Performing silhouette analysis for {min_clusters} to {max_clusters} clusters")
    
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = {}
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        start_time = time.time()
        
        # Perform hierarchical clustering
        Z = linkage(sample_features, method='ward')
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Ensure we have at least 2 clusters (silhouette score requires at least 2)
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) < 2:
            logger.warning(f"Only found {len(unique_clusters)} unique clusters for n_clusters={n_clusters}, skipping")
            continue
        
        # Calculate silhouette score
        try:
            score = silhouette_score(sample_features, cluster_labels)
            silhouette_scores[n_clusters] = score
            
            logger.debug(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}, Time: {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.warning(f"Error calculating silhouette score for {n_clusters} clusters: {str(e)}")
    
    # Find optimal number of clusters
    if not silhouette_scores:
        logger.warning("No valid silhouette scores found, defaulting to min_clusters")
        return min_clusters, {}
    
    optimal_clusters = max(silhouette_scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Optimal number of clusters: {optimal_clusters} with score {silhouette_scores[optimal_clusters]:.4f}")
    
    return optimal_clusters, silhouette_scores

def generate_final_cluster_visualization(
                               final_clusters: Dict[str, List[str]],
                               container_features: Dict[str, Tuple[float, float, int]],
                               output_path: str,
                               use_distinct_aisles: bool = True,
                               critical_containers: Optional[List[str]] = None) -> None:
    """
    Generate a scatter plot showing all final clusters after hierarchical sub-clustering
    
    Parameters
    ----------
    final_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_features : Dict[str, Tuple[float, float, int]]
        Dictionary mapping container IDs to features (centroid, span, distinct_aisles)
    output_path : str
        Path to save visualizations
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature
    critical_containers : Optional[List[str]]
        List of critical container IDs to mark with 'x' markers
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    
    plt.figure(figsize=(16, 10))
    
    # Create a set of critical containers for faster lookups
    critical_set = set() if critical_containers is None else set(critical_containers)
    
    # Create a colormap with enough colors for all clusters
    num_clusters = len(final_clusters)
    if num_clusters > 0:
        # Use a qualitative colormap for discrete colors
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        
        # Create a mapping from original cluster IDs to sequential integers for coloring
        cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(final_clusters.keys())}
        
        # Prepare data per cluster for legend and scatter plot
        cluster_data = {}
        
        for cluster_id, container_ids in final_clusters.items():
            cluster_idx = cluster_id_to_idx[cluster_id]
            color = cmap(cluster_idx)
            
            # Initialize data structures for this cluster
            cluster_data[cluster_id] = {
                'centroids': [],
                'secondary_features': [],
                'is_critical': [],
                'color': color
            }
            
            # Process containers
            for container_id in container_ids:
                if container_id in container_features:
                    centroid, span, distinct_aisles = container_features[container_id]
                    is_critical = container_id in critical_set
                    
                    cluster_data[cluster_id]['centroids'].append(centroid)
                    
                    # Choose the appropriate secondary feature
                    if use_distinct_aisles:
                        cluster_data[cluster_id]['secondary_features'].append(distinct_aisles)
                    else:
                        cluster_data[cluster_id]['secondary_features'].append(span)
                    
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
                    [data['secondary_features'][i] for i in range(len(data['secondary_features'])) if non_critical_mask[i]],
                    color=data['color'],
                    marker='o',
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            # Plot critical containers with 'x' marker
            critical_mask = data['is_critical']
            if any(critical_mask):
                plt.scatter(
                    [data['centroids'][i] for i in range(len(data['centroids'])) if critical_mask[i]],
                    [data['secondary_features'][i] for i in range(len(data['secondary_features'])) if critical_mask[i]],
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
            avg_secondary = sum(data['secondary_features']) / len(data['secondary_features'])
            
            # Annotate cluster ID at cluster centroid
            plt.annotate(
                cluster_id,
                (avg_centroid, avg_secondary),
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Add markers to legend for critical and non-critical
        from matplotlib.lines import Line2D
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
        plt.ylabel('Distinct Aisles' if use_distinct_aisles else 'Aisle Span', fontsize=12)
        plt.title(f'Final Clusters After Hierarchical Sub-clustering ({num_clusters} clusters)', fontsize=14)
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
        
        # Make sure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'final_clusters_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional plot with cluster statistics
        plt.figure(figsize=(14, 8))
        
        # Calculate statistics per cluster
        cluster_sizes = []
        critical_percentages = []
        cluster_labels = []
        
        for cluster_id, data in cluster_data.items():
            if not data['centroids']:
                continue
                
            total_containers = len(data['centroids'])
            critical_count = sum(data['is_critical'])
            critical_pct = 100 * critical_count / total_containers if total_containers else 0
            
            cluster_sizes.append(total_containers)
            critical_percentages.append(critical_pct)
            cluster_labels.append(cluster_id)
        
        # Create bar chart with dual y-axis
        x = range(len(cluster_labels))
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot container count bars
        bars = ax1.bar(x, cluster_sizes, color=[cluster_data[label]['color'] for label in cluster_labels], alpha=0.7)
        ax1.set_xlabel('Cluster ID', fontsize=12)
        ax1.set_ylabel('Container Count', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cluster_labels, rotation=45)
        
        # Create second y-axis for critical percentage
        ax2 = ax1.twinx()
        ax2.plot(x, critical_percentages, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Critical Container Percentage (%)', color='r', fontsize=12)
        ax2.tick_params(axis='y', colors='r')
        ax2.set_ylim(0, 100)
        
        # Add value labels to bars
        for i, v in enumerate(cluster_sizes):
            ax1.text(i, v + 0.5, str(v), ha='center', fontsize=9)
            ax2.text(i, critical_percentages[i] + 2, f"{critical_percentages[i]:.1f}%", 
                    ha='center', color='darkred', fontsize=9)
        
        plt.title('Cluster Sizes and Critical Container Percentages', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'cluster_statistics.png'), dpi=300)
        plt.close()
        