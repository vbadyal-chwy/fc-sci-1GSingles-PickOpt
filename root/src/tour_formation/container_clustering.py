import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import time

logger = logging.getLogger(__name__)

'''def get_container_aisles(container_id: str, container_data: pd.DataFrame, 
                         sku_aisle_mapping: Dict[str, List[int]]) -> Set[int]:
    """Get all aisles required for a specific container"""
    container_skus = container_data[container_data['container_id'] == container_id]['item_number'].unique()
    
    all_aisles = set()
    for sku in container_skus:
        if sku in sku_aisle_mapping:
            all_aisles.update(sku_aisle_mapping[sku])
    
    return all_aisles'''

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

def determine_optimal_clusters(feature_matrix: np.ndarray, 
                              min_clusters: int = 2, 
                              max_clusters: int = 10,
                              max_sample_size: int = 1000) -> Tuple[int, Dict[int, float]]:
    """
    Determine optimal number of clusters using silhouette analysis
    
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
    # If we have too many containers, sample to improve performance
    '''if feature_matrix.shape[0] > max_sample_size:
        logger.info(f"Sampling {max_sample_size} containers for silhouette analysis")
        indices = np.random.choice(feature_matrix.shape[0], max_sample_size, replace=False)
        sample_features = feature_matrix[indices]
    else:'''
    sample_features = feature_matrix
    
    # Adjust max_clusters based on data size
    max_clusters = min(max_clusters, sample_features.shape[0] // 5)
    max_clusters = max(min_clusters + 1, max_clusters)  # Ensure at least 2 values to compare
    
    logger.info(f"Performing silhouette analysis for {min_clusters} to {max_clusters} clusters")
    
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
        score = silhouette_score(sample_features, cluster_labels)
        silhouette_scores[n_clusters] = score
        
        logger.info(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}, Time: {time.time() - start_time:.2f}s")
    
    # Find optimal number of clusters
    if not silhouette_scores:
        logger.warning("No valid silhouette scores found, defaulting to min_clusters")
        return min_clusters, {}
    
    optimal_clusters = max(silhouette_scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Optimal number of clusters: {optimal_clusters} with score {silhouette_scores[optimal_clusters]:.4f}")
    
    return optimal_clusters, silhouette_scores

def preprocess_container_features(container_data: pd.DataFrame, 
                                 slotbook_data: pd.DataFrame, 
                                 container_ids: List[str],
                                 sku_aisle_mapping: Dict[str, List[int]],
                                 use_distinct_aisles: bool = True,
                                 centroid_weight: float = 0.5,
                                 secondary_weight: float = 0.5) -> Tuple[Dict[str, Tuple[float, float, int]], 
                                                                      List[List[float]], 
                                                                      List[str], 
                                                                      np.ndarray]:
    """
    Extract and preprocess features for container clustering
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with SKU information
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    container_ids : List[str]
        List of container IDs to process
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature
    centroid_weight : float
        Weight for the centroid feature
    secondary_weight : float
        Weight for the secondary feature
        
    Returns
    -------
    Tuple[Dict[str, Tuple[float, float, int]], List[List[float]], List[str], np.ndarray]
        - Dictionary mapping container IDs to features
        - List of feature arrays for valid containers
        - List of valid container IDs
        - Normalized feature matrix
    """
    # Compute features for each container
    container_features = {}
    feature_arrays = []
    valid_containers = []
    
    for container_id in container_ids:
        centroid, span, distinct_aisles = compute_container_features(container_id, container_data, sku_aisle_mapping)
        container_features[container_id] = (centroid, span, distinct_aisles)
        
        # Skip containers with no aisle data
        if centroid != 0 or span != 0:
            # Choose secondary feature based on flag
            secondary_feature = distinct_aisles if use_distinct_aisles else span
            
            feature_arrays.append([
                centroid * centroid_weight,           # Aisle centroid 
                secondary_feature * secondary_weight  # Secondary feature (either distinct aisles or span)
            ])
            valid_containers.append(container_id)
    
    if not valid_containers:
        return container_features, feature_arrays, valid_containers, np.array([])
    
    # Create and normalize feature matrix
    feature_matrix = np.array(feature_arrays)
    
    # Normalize features if we have data
    if feature_matrix.size > 0:
        for col in range(feature_matrix.shape[1]):
            col_range = np.max(feature_matrix[:, col]) - np.min(feature_matrix[:, col])
            if col_range > 0:
                feature_matrix[:, col] = (feature_matrix[:, col] - np.min(feature_matrix[:, col])) / col_range
    
    return container_features, feature_arrays, valid_containers, feature_matrix

def hierarchical_sub_clustering(container_data: pd.DataFrame, 
                               slotbook_data: pd.DataFrame,
                               container_ids: List[str],
                               sku_aisle_mapping: Dict[str, List[int]],
                               parent_cluster_id: str,
                               max_cluster_size: int,
                               use_distinct_aisles: bool,
                               min_clusters: int,
                               max_clusters: int,
                               depth: int = 0,
                               centroid_weight: float = 0.5,
                               secondary_weight: float = 0.5,
                               generate_visuals: bool = False,
                               output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Apply hierarchical sub-clustering to a set of containers
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with SKU information
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    container_ids : List[str]
        List of container IDs to cluster
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
    parent_cluster_id : str
        ID of the parent cluster
    max_cluster_size : int
        Maximum allowed cluster size
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature
    min_clusters : int
        Minimum number of clusters to consider
    max_clusters : int
        Maximum number of clusters to consider
    depth : int
        Current recursion depth
    centroid_weight : float
        Weight for centroid feature
    secondary_weight : float
        Weight for secondary feature
    generate_visuals : bool
        Whether to generate visualizations
    output_path : str
        Path to save visualizations
        
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
    
    # Adjust feature weights at each level to focus on different aspects
    # Deeper levels put more emphasis on the secondary feature
    adjusted_centroid_weight = max(0.3, centroid_weight - (depth * 0.1))
    adjusted_secondary_weight = min(0.7, secondary_weight + (depth * 0.1))
    
    # Preprocess features
    container_features, feature_arrays, valid_containers, feature_matrix = preprocess_container_features(
        container_data, slotbook_data, container_ids, sku_aisle_mapping, 
        use_distinct_aisles, adjusted_centroid_weight, adjusted_secondary_weight
    )
    
    # If no valid containers or too few for clustering, return as is
    if len(valid_containers) < min_clusters:
        return {f"{parent_cluster_id}": container_ids}
    
    # Determine optimal number of clusters for this level
    # Use number of clusters that would make each cluster close to max_cluster_size
    estimated_optimal_clusters = max(min_clusters, len(valid_containers) // max_cluster_size + 1)
    max_possible_clusters = min(max_clusters, len(valid_containers) // 2)  # Avoid too many clusters
    
    '''# If we have a very large cluster, use the estimate directly
    if len(valid_containers) > 5 * max_cluster_size:
        optimal_clusters = min(estimated_optimal_clusters, max_possible_clusters)
        logger.info(f"{indent}Using estimated optimal clusters: {optimal_clusters} for large cluster")
        silhouette_scores = {}
    else:'''
    # For smaller clusters, use silhouette analysis but cap at our estimate
    optimal_clusters, silhouette_scores = determine_optimal_clusters(
        feature_matrix, 
        min_clusters=min_clusters,
        max_clusters=min(max_possible_clusters, estimated_optimal_clusters + 2)
    )
    
    # Perform hierarchical clustering with optimal number of clusters
    Z = linkage(feature_matrix, method='ward')
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to sub-clusters
    sub_clusters = defaultdict(list)
    for i, container_id in enumerate(valid_containers):
        sub_cluster_id = int(cluster_labels[i])
        sub_clusters[sub_cluster_id].append(container_id)
    
    # Generate visualizations for this level if requested
    if generate_visuals and depth < 3:  # Limit visualization depth
        sub_cluster_path = os.path.join(output_path, f"cluster_{parent_cluster_id}_depth_{depth}")
        os.makedirs(sub_cluster_path, exist_ok=True)
        
        try:
            generate_cluster_visualizations(
                feature_matrix, cluster_labels, container_features, 
                valid_containers, Z, silhouette_scores, sub_cluster_path, use_distinct_aisles,
                title_prefix=f"Cluster {parent_cluster_id} (Depth {depth})"
            )
        except Exception as e:
            logger.warning(f"Failed to generate sub-cluster visualizations: {str(e)}")
    
    # Check each sub-cluster for size and recurse if needed
    final_clusters = {}
    for sub_id, containers in sub_clusters.items():
        sub_cluster_id = f"{parent_cluster_id}_{sub_id}"
        
        # Log sub-cluster statistics
        logger.info(f"{indent}Created sub-cluster {sub_cluster_id} with {len(containers)} containers")
        
        # Recursively apply clustering if still too large
        if len(containers) > max_cluster_size:
            logger.info(f"{indent}Sub-cluster {sub_cluster_id} exceeds max size, further sub-clustering...")
            next_level_clusters = hierarchical_sub_clustering(
                container_data, slotbook_data, containers, sku_aisle_mapping,
                sub_cluster_id, max_cluster_size, use_distinct_aisles,
                min_clusters, max_clusters, depth + 1,
                adjusted_centroid_weight, adjusted_secondary_weight,
                generate_visuals, output_path
            )
            final_clusters.update(next_level_clusters)
        else:
            final_clusters[sub_cluster_id] = containers
    
    return final_clusters

def cluster_containers(container_data: pd.DataFrame, slotbook_data: pd.DataFrame, 
                      batch_size: Optional[int] = None, 
                      containers_per_tour: int = 2,
                      max_cluster_size: int = 500,
                      use_distinct_aisles: bool = True, 
                      min_clusters: int = 2, 
                      max_clusters: int = 10,
                      generate_visuals: bool = False, 
                      output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Cluster containers by aisle requirements with hierarchical sub-clustering for large clusters
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    batch_size : Optional[int]
        Optional number of tours to form in each batch (if None, determined automatically)
    containers_per_tour : int
        Maximum containers per tour
    max_cluster_size : int
        Maximum size for any cluster, larger clusters will be further sub-clustered
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature
    min_clusters : int
        Minimum number of clusters to consider
    max_clusters : int
        Maximum number of clusters to consider
    generate_visuals : bool
        Whether to generate visualizations of the clustering
    output_path : str
        Path to save visualizations if generated
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    logger.info("Clustering containers with hierarchical sub-clustering")
    logger.info(f"Using {'distinct_aisles' if use_distinct_aisles else 'aisle_span'} as secondary feature")
    logger.info(f"Maximum cluster size set to {max_cluster_size} containers")
    
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
           
    # Get unique container IDs
    container_ids = container_data['container_id'].unique().tolist()
    
    # Preprocess features
    container_features, feature_arrays, valid_containers, feature_matrix = preprocess_container_features(
        container_data, slotbook_data, container_ids, sku_aisle_mapping, use_distinct_aisles
    )
    
    if not valid_containers:
        logger.warning("No valid containers found with aisle data")
        return {"0": []}
    
    # Determine optimal number of clusters for top level
    if batch_size is None:
        optimal_clusters, silhouette_scores = determine_optimal_clusters(
            feature_matrix, min_clusters, max_clusters
        )
    else:
        # If batch_size is provided, calculate target number of clusters based on it
        target_cluster_size = batch_size * containers_per_tour
        optimal_clusters = max(min_clusters, len(valid_containers) // target_cluster_size)
        logger.info(f"Using batch_size parameter to set clusters to: {optimal_clusters}")
        silhouette_scores = {}  # Empty dict as we're not computing scores
    
    # Perform hierarchical clustering with optimal number of clusters
    Z = linkage(feature_matrix, method='ward')
    cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
    
    # Map containers to initial top-level clusters
    initial_clusters = defaultdict(list)
    
    for i, container_id in enumerate(valid_containers):
        cluster_id = int(cluster_labels[i])
        initial_clusters[cluster_id].append(container_id)
    
    # Generate visualizations for top level clusters
    if generate_visuals:
        try:
            generate_cluster_visualizations(
                feature_matrix, cluster_labels, container_features, 
                valid_containers, Z, silhouette_scores, output_path, use_distinct_aisles,
                title_prefix="Top Level Clustering"
            )
        except Exception as e:
            logger.warning(f"Failed to generate cluster visualizations: {str(e)}")
    
    # Apply hierarchical sub-clustering to any oversized clusters
    final_clusters = {}
    for cluster_id, containers in initial_clusters.items():
        # Check if this cluster needs sub-clustering
        if len(containers) > max_cluster_size:
            logger.info(f"Cluster {cluster_id} exceeds max size with {len(containers)} containers, applying sub-clustering...")
            sub_clusters = hierarchical_sub_clustering(
                container_data, slotbook_data, containers, sku_aisle_mapping,
                str(cluster_id), max_cluster_size, use_distinct_aisles,
                min_clusters, max_clusters, depth=1,
                generate_visuals=generate_visuals, output_path=output_path
            )
            final_clusters.update(sub_clusters)
        else:
            final_clusters[str(cluster_id)] = containers
    
    # Sort clusters by their centroid value for easier processing
    sorted_clusters = {}
    cluster_centroids = {}
    
    for cluster_id, cluster_containers in final_clusters.items():
        # Calculate average centroid for this cluster
        centroids = [container_features[c_id][0] for c_id in cluster_containers if c_id in container_features]
        if centroids:
            avg_centroid = np.mean(centroids)
            cluster_centroids[cluster_id] = avg_centroid
    
    # Create final sorted clusters with sequential IDs
    for i, (cluster_id, _) in enumerate(sorted(cluster_centroids.items(), key=lambda x: x[1])):
        sorted_clusters[str(i)] = final_clusters[cluster_id]
    
    # Add containers that didn't make it into valid_containers (if any)
    missing_containers = set(container_ids) - set(valid_containers)
    if missing_containers:
        logger.warning(f"Found {len(missing_containers)} containers with no valid aisle data")
        if sorted_clusters:
            # Add to the first cluster
            first_key = next(iter(sorted_clusters))
            sorted_clusters[first_key].extend(list(missing_containers))
        else:
            # Create a special cluster for these
            sorted_clusters["missing_data"] = list(missing_containers)
    
    # Calculate and log cluster statistics
    cluster_stats = calculate_cluster_statistics(sorted_clusters, container_features, use_distinct_aisles)
    
    # Log clustering results
    logger.info(f"Created {len(sorted_clusters)} final clusters from {len(valid_containers)} containers")
    logger.info("\nFinal Cluster Statistics:")
    for cluster_id, stats in cluster_stats.items():
        if use_distinct_aisles:
            logger.info(f"Cluster {cluster_id}: {stats['size']} containers, "
                        f"Centroid: {stats['avg_centroid']:.1f}, "
                        f"Distinct Aisles: {stats['avg_distinct_aisles']:.1f}, "
                        f"Cohesion: {stats['cohesion']:.3f}")
        else:
            logger.info(f"Cluster {cluster_id}: {stats['size']} containers, "
                        f"Centroid: {stats['avg_centroid']:.1f}, "
                        f"Span: {stats['avg_span']:.1f}, "
                        f"Cohesion: {stats['cohesion']:.3f}")
    
    # Generate final cluster visualization
    if generate_visuals:
        try:
            generate_final_cluster_visualization(sorted_clusters, container_features, output_path, use_distinct_aisles)
            logger.info(f"Generated final cluster visualization in {output_path}")
        except Exception as e:
            logger.warning(f"Failed to generate final cluster visualization: {str(e)}")
    
    # Write container details to CSV
    write_cluster_details_to_csv(sorted_clusters, container_features, valid_containers, container_ids, output_path)
    
    return sorted_clusters

def write_cluster_details_to_csv(sorted_clusters: Dict[str, List[str]],
                               container_features: Dict[str, Tuple[float, float, int]],
                               valid_containers: List[str],
                               all_container_ids: List[str],
                               output_path: str) -> None:
    """Write cluster details to CSV file"""
    os.makedirs(output_path, exist_ok=True)
    
    # Create a mapping from container_id to cluster_id
    container_to_cluster = {}
    for cluster_id, container_list in sorted_clusters.items():
        for container_id in container_list:
            container_to_cluster[container_id] = cluster_id
    
    # Create a dataframe with container details
    container_details = []
    for container_id in all_container_ids:
        if container_id in container_features:
            centroid, span, distinct_aisles = container_features[container_id]
            cluster = container_to_cluster.get(container_id, "missing")
            container_details.append({
                'container_id': container_id,
                'aisle_centroid': centroid,
                'aisle_span': span,
                'distinct_aisles': distinct_aisles,
                'cluster': cluster
            })
        else:
            # For containers with no feature data
            container_details.append({
                'container_id': container_id,
                'aisle_centroid': np.nan,
                'aisle_span': np.nan,
                'distinct_aisles': np.nan,
                'cluster': container_to_cluster.get(container_id, "missing")
            })
    
    # Convert to dataframe and save to CSV
    if container_details:
        container_df = pd.DataFrame(container_details)
        container_df.to_csv(os.path.join(output_path, 'container_clusters.csv'), index=False)
        logger.info(f"Container details saved to {os.path.join(output_path, 'container_clusters.csv')}")

def calculate_cluster_statistics(clusters: Dict[str, List[str]], 
                               container_features: Dict[str, Tuple[float, float, int]],
                               use_distinct_aisles: bool = True) -> Dict[str, Dict[str, float]]:
    """Calculate detailed statistics for each cluster"""
    stats = {}
    
    for cluster_id, container_ids in clusters.items():
        # Extract features for containers in this cluster that have feature data
        valid_containers = [c_id for c_id in container_ids if c_id in container_features]
        
        if not valid_containers:
            # Handle clusters with no valid container features
            stats[cluster_id] = {
                'size': len(container_ids),
                'avg_centroid': np.nan,
                'avg_span': np.nan,
                'avg_distinct_aisles': np.nan,
                'centroid_std': np.nan,
                'span_std': np.nan,
                'distinct_aisles_std': np.nan,
                'cohesion': np.nan
            }
            continue
            
        centroids = [container_features[c_id][0] for c_id in valid_containers]
        spans = [container_features[c_id][1] for c_id in valid_containers]
        distinct_aisles = [container_features[c_id][2] for c_id in valid_containers]
        
        # Calculate cluster statistics
        avg_centroid = np.mean(centroids) if centroids else 0
        avg_span = np.mean(spans) if spans else 0
        avg_distinct_aisles = np.mean(distinct_aisles) if distinct_aisles else 0
        
        # Calculate cohesion (lower value = more cohesive cluster)
        # Measures the average distance of container centroids from cluster centroid
        centroid_diffs = [abs(c - avg_centroid) for c in centroids]
        cohesion = np.mean(centroid_diffs) if centroid_diffs else 0
        
        stats[cluster_id] = {
            'size': len(container_ids),
            'avg_centroid': avg_centroid,
            'avg_span': avg_span,
            'avg_distinct_aisles': avg_distinct_aisles,
            'centroid_std': np.std(centroids) if len(centroids) > 1 else 0,
            'span_std': np.std(spans) if len(spans) > 1 else 0,
            'distinct_aisles_std': np.std(distinct_aisles) if len(distinct_aisles) > 1 else 0,
            'cohesion': cohesion
        }
    
    return stats

def generate_final_cluster_visualization(
                               final_clusters: Dict[str, List[str]],
                               container_features: Dict[str, Tuple[float, float, int]],
                               output_path: str,
                               use_distinct_aisles: bool = True) -> None:
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
    """
    plt.figure(figsize=(14, 10))
    
    # Create a colormap with enough colors for all clusters
    num_clusters = len(final_clusters)
    if num_clusters > 0:
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        
        # Create a mapping from original cluster IDs to sequential integers for coloring
        cluster_id_to_idx = {cluster_id: i for i, cluster_id in enumerate(final_clusters.keys())}
        
        # Extract centroids and secondary features for all containers
        centroids = []
        secondary_features = []
        colors = []
        annotations = []
        
        for cluster_id, container_ids in final_clusters.items():
            for container_id in container_ids:
                if container_id in container_features:
                    centroid, span, distinct_aisles = container_features[container_id]
                    
                    centroids.append(centroid)
                    
                    # Choose the appropriate secondary feature
                    if use_distinct_aisles:
                        secondary_features.append(distinct_aisles)
                    else:
                        secondary_features.append(span)
                    
                    colors.append(cmap(cluster_id_to_idx[cluster_id]))
                    annotations.append(cluster_id)
        
        # Create scatter plot
        scatter = plt.scatter(centroids, secondary_features, c=colors, alpha=0.7, s=30)
        
        # Add a few annotations for cluster centers
        cluster_centers = {}
        for i, (centroid, secondary, annotation) in enumerate(zip(centroids, secondary_features, annotations)):
            if annotation not in cluster_centers:
                cluster_centers[annotation] = (centroid, secondary)
        
        # Limit the number of annotations to avoid clutter
        max_annotations = min(20, len(cluster_centers))
        selected_centers = dict(list(cluster_centers.items())[:max_annotations])
        
        for annotation, (x, y) in selected_centers.items():
            plt.annotate(annotation, (x, y), fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add labels and title
        plt.xlabel('Aisle Centroid')
        plt.ylabel('Distinct Aisles' if use_distinct_aisles else 'Aisle Span')
        plt.title(f'Final Clusters After Hierarchical Sub-clustering ({num_clusters} clusters)')
        plt.grid(True, alpha=0.3)
        
        # Add color bar with cluster count
        cbar = plt.colorbar(scatter, ticks=[])
        cbar.set_label(f'{num_clusters} Clusters')
        
        # Save the figure
        plt.savefig(os.path.join(output_path, 'final_clusters_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_cluster_visualizations(feature_matrix: np.ndarray, 
                                  cluster_labels: np.ndarray,
                                  container_features: Dict[str, Tuple[float, float, int]],
                                  valid_containers: List[str],
                                  linkage_matrix: np.ndarray,
                                  silhouette_scores: Dict[int, float],
                                  output_path: str,
                                  use_distinct_aisles: bool = True,
                                  title_prefix: str = "") -> None:
    """Generate visualizations for cluster analysis"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Use a built-in colormap 
    cmap = plt.cm.get_cmap('tab20', len(np.unique(cluster_labels)))
    
    # 1. Scatter plot of containers by aisle centroid and secondary feature
    plt.figure(figsize=(12, 8))
    raw_centroids = [container_features[c][0] for c in valid_containers]
    
    if use_distinct_aisles:
        secondary_feature = [container_features[c][2] for c in valid_containers]
        secondary_label = 'Distinct Aisles'
    else:
        secondary_feature = [container_features[c][1] for c in valid_containers]
        secondary_label = 'Aisle Span'
    
    plt.scatter(raw_centroids, secondary_feature, c=cluster_labels, cmap=cmap, alpha=0.7, s=50)
    plt.colorbar(label='Cluster')
    plt.xlabel('Aisle Centroid')
    plt.ylabel(secondary_label)
    title = 'Container Clustering by Aisle Centroid and ' + secondary_label
    if title_prefix:
        title = f"{title_prefix}: {title}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'cluster_scatter.png'), dpi=300)
    plt.close()
    
    # 6. 2D PCA visualization (if we have enough features)
    if feature_matrix.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(feature_matrix)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap=cmap, alpha=0.7, s=50)
        plt.colorbar(label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        title = 'PCA of Container Features'
        if title_prefix:
            title = f"{title_prefix}: {title}"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'cluster_pca.png'), dpi=300)
        plt.close()
    
    # 7. Silhouette analysis visualization for the optimal number of clusters
    if feature_matrix.shape[0] <= 5000:  # Only for reasonably sized datasets
        try:
            # Get the optimal number of clusters if available
            if silhouette_scores:
                n_clusters = max(silhouette_scores.items(), key=lambda x: x[1])[0]
            else:
                n_clusters = len(np.unique(cluster_labels))
            
            # Calculate silhouette values
            silhouette_vals = silhouette_samples(feature_matrix, cluster_labels)
            
            plt.figure(figsize=(12, 8))
            y_lower, y_upper = 0, 0
            
            for i, cluster in enumerate(np.unique(cluster_labels)):
                cluster_silhouette_vals = silhouette_vals[cluster_labels == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                
                color = cmap(i / len(np.unique(cluster_labels)))
                plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                        height=1.0, edgecolor='none', color=color)
                
                plt.text(-0.05, (y_lower + y_upper) / 2, str(i + 1))
                y_lower += len(cluster_silhouette_vals)
            
            # Add vertical line for average silhouette score
            plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
            title = f"Silhouette Analysis for {n_clusters} Clusters"
            if title_prefix:
                title = f"{title_prefix}: {title}"
            plt.title(title)
            plt.xlabel("Silhouette Coefficient")
            plt.ylabel("Cluster")
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'silhouette_analysis.png'), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate silhouette analysis visualization: {str(e)}")
    
    # 8. Generate summary statistics table
    cluster_stats = {}
    for cluster_id in np.unique(cluster_labels):
        # Convert cluster_id to int
        int_cluster_id = int(cluster_id)
        mask = cluster_labels == cluster_id
        cluster_data = feature_matrix[mask]
        cluster_containers = np.array(valid_containers)[mask]
        
        cluster_stats[int_cluster_id] = {
            'size': len(cluster_containers),
            'centroid_mean': np.mean([container_features[c][0] for c in cluster_containers]),
            'span_mean': np.mean([container_features[c][1] for c in cluster_containers]),
            'distinct_aisles_mean': np.mean([container_features[c][2] for c in cluster_containers]),
            'centroid_std': np.std([container_features[c][0] for c in cluster_containers]),
            'span_std': np.std([container_features[c][1] for c in cluster_containers]),
            'distinct_aisles_std': np.std([container_features[c][2] for c in cluster_containers]),
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(cluster_stats).T
    stats_df.index.name = 'Cluster'
    stats_df.to_csv(os.path.join(output_path, 'cluster_statistics.csv'))
    
    # 2. Dendrogram visualization
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    title = 'Hierarchical Clustering Dendrogram'
    if title_prefix:
        title = f"{title_prefix}: {title}"
    plt.title(title)
    plt.xlabel('Container Index')
    plt.ylabel('Distance')
    plt.savefig(os.path.join(output_path, 'cluster_dendrogram.png'), dpi=300)
    plt.close()
    
    # 3. Cluster sizes bar chart
    # Convert cluster labels to integers to avoid warnings
    cluster_sizes = pd.Series(cluster_labels).astype(int).value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar', color='skyblue')
    plt.axhline(y=np.mean(cluster_sizes), color='r', linestyle='--', label='Average Size')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Containers')
    title = 'Cluster Size Distribution'
    if title_prefix:
        title = f"{title_prefix}: {title}"
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cluster_sizes.png'), dpi=300)
    plt.close()
    
    # 4. Silhouette score plot (if scores are provided)
    if silhouette_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'o-', color='blue')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        title = 'Silhouette Score by Number of Clusters'
        if title_prefix:
            title = f"{title_prefix}: {title}"
        plt.title(title)
        
        # Highlight the optimal number of clusters
        optimal_clusters = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        optimal_score = silhouette_scores[optimal_clusters]
        plt.plot(optimal_clusters, optimal_score, 'o', color='red', markersize=10)
        plt.annotate(f'Optimal: {optimal_clusters} clusters',
                    xy=(optimal_clusters, optimal_score),
                    xytext=(optimal_clusters + 0.5, optimal_score - 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.savefig(os.path.join(output_path, 'silhouette_scores.png'), dpi=300)
        plt.close()
    
    # 5. Feature distribution by cluster
    # Extract raw features for plotting and ensure cluster labels are integers
    feature_dict = {
        'Container': valid_containers,
        'Cluster': cluster_labels.astype(int),  # Convert to int explicitly
        'Centroid': [container_features[c][0] for c in valid_containers],
        'Span': [container_features[c][1] for c in valid_containers],
        'DistinctAisles': [container_features[c][2] for c in valid_containers]
    }
    
    df_features = pd.DataFrame(feature_dict)
    
    # Box plots for features by cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.boxplot(x='Cluster', y='Centroid', data=df_features, ax=axes[0])
    axes[0].set_title('Aisle Centroid Distribution by Cluster')
    axes[0].grid(True, alpha=0.3)
    
    sns.boxplot(x='Cluster', y='Span', data=df_features, ax=axes[1])
    axes[1].set_title('Aisle Span Distribution by Cluster')
    axes[1].grid(True, alpha=0.3)
    
    sns.boxplot(x='Cluster', y='DistinctAisles', data=df_features, ax=axes[2])
    axes[2].set_title('Distinct Aisles Distribution by Cluster')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cluster_features.png'), dpi=300)
    plt.close()