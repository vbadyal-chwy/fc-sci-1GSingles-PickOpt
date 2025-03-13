import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

logger = logging.getLogger(__name__)

def select_optimal_clusters(
    all_clusters: Dict[str, List[str]],
    container_data: pd.DataFrame,
    max_picking_capacity: int,
    config: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Select optimal clusters for tour formation ensuring all critical containers are included.
    
    Parameters
    ----------
    all_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_data : pd.DataFrame
        Container data with order details and slack categories
    max_picking_capacity : int
        Maximum number of containers for picking
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Dict[str, List[str]]
        Selected clusters mapping cluster IDs to lists of container IDs
    """
    logger.info("Starting optimal cluster selection")
    
    # PHASE 1: Guarantee critical container coverage
    
    # Identify critical and urgent containers
    critical_containers = get_critical_containers(container_data)
    logger.info(f"Found {len(critical_containers)} critical/urgent containers")
    
    # Step 1: Identify and select clusters with critical containers (up to capacity limit)
    selected_clusters, critical_coverage = select_clusters_with_critical_containers(
        all_clusters, critical_containers, max_picking_capacity
    )
    
    # Step 2: Handle critical containers not yet covered
    uncovered_critical = critical_containers - critical_coverage
    if uncovered_critical:
        logger.info(f"{len(uncovered_critical)} critical containers not covered by initial selection")
        selected_clusters, critical_coverage = handle_uncovered_critical_containers(
            uncovered_critical, selected_clusters, container_data, all_clusters
        )
    
    # Verify all critical containers are covered
    if critical_coverage != critical_containers:
        remaining = critical_containers - critical_coverage
        logger.warning(f"Failed to cover all critical containers. {len(remaining)} containers remain uncovered.")
    else:
        logger.info("All critical containers are covered")
    
    # PHASE 2: Fill remaining capacity with optimal clusters using greedy approach
    
    # Calculate current capacity utilization
    current_container_count = count_containers_in_clusters(selected_clusters)
    logger.info(f"Initial selection: {len(selected_clusters)} clusters with {current_container_count} containers")
    
    # Score and rank remaining eligible clusters
    if current_container_count < max_picking_capacity:
        remaining_clusters = {
            cluster_id: containers for cluster_id, containers in all_clusters.items()
            if cluster_id not in selected_clusters
        }
        
        if remaining_clusters:
            # Add highest scoring clusters until we reach capacity
            scored_remaining = score_clusters(remaining_clusters, container_data)
            additional_clusters = greedy_cluster_selection(
                scored_remaining, 
                max_picking_capacity - current_container_count
            )
            
            # Add additional clusters to selected
            for cluster_id, containers in additional_clusters.items():
                selected_clusters[cluster_id] = containers
    
    # Final validation
    final_container_count = count_containers_in_clusters(selected_clusters)
    logger.info(f"Final selection: {len(selected_clusters)} clusters with {final_container_count} containers")
    
    return selected_clusters

def get_critical_containers(container_data: pd.DataFrame) -> Set[str]:
    """
    Extract critical and urgent containers from container data.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with slack_category column
        
    Returns
    -------
    Set[str]
        Set of critical and urgent container IDs
    """
    if 'slack_category' not in container_data.columns:
        logger.warning("No slack_category column found in container data")
        return set()
    
    # Get unique container IDs with Critical or Urgent status
    critical_urgent = container_data[
        container_data['slack_category'].isin(['Critical', 'Urgent'])
    ]['container_id'].unique()
    
    return set(critical_urgent)

def select_clusters_with_critical_containers(
    all_clusters: Dict[str, List[str]],
    critical_containers: Set[str],
    max_picking_capacity: int
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Select clusters that contain critical containers using a greedy approach.
    Prioritizes clusters with the most critical containers until max_picking_capacity is reached.
    
    Parameters
    ----------
    all_clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : Set[str]
        Set of critical container IDs
    max_picking_capacity : int
        Maximum number of containers for picking
        
    Returns
    -------
    Tuple[Dict[str, List[str]], Set[str]]
        - Selected clusters (cluster_id -> container_ids)
        - Set of covered critical container IDs
    """
    selected_clusters = {}
    critical_coverage = set()
    total_containers = 0
    
    # Score clusters by critical container count
    scored_clusters = []
    for cluster_id, container_ids in all_clusters.items():
        # Find intersection with critical containers
        critical_in_cluster = set(container_ids).intersection(critical_containers)
        
        if critical_in_cluster:
            scored_clusters.append({
                'cluster_id': cluster_id,
                'containers': container_ids,
                'critical_count': len(critical_in_cluster),
                'size': len(container_ids),
                'critical_containers': critical_in_cluster
            })
    
    # Sort by critical container count (descending)
    scored_clusters.sort(key=lambda x: x['critical_count'], reverse=True)
    
    # Greedily select clusters until we reach capacity
    for cluster in scored_clusters:
        # Check if adding this cluster would exceed capacity
        if total_containers + cluster['size'] > max_picking_capacity:
            # Skip if we're already at or over capacity
            if total_containers >= max_picking_capacity:
                logger.info(f"Stopping critical cluster selection: reached capacity ({total_containers} containers)")
                break
            
            # Otherwise, assess whether this critical cluster is important enough to include
            # despite exceeding capacity
            remaining_critical = len(critical_containers) - len(critical_coverage)
            coverage_gain = len(cluster['critical_containers'] - critical_coverage)
            
            # Skip if the gain is marginal compared to remaining critical containers
            if coverage_gain < remaining_critical * 0.0:  # Skip if covering less than 20% of remaining
                logger.info(f"Skipping critical cluster: marginal gain ({coverage_gain} critical containers)")
                continue
        
        # Add this cluster
        selected_clusters[cluster['cluster_id']] = cluster['containers']
        critical_coverage.update(cluster['critical_containers'])
        total_containers += cluster['size']
        
        logger.info(f"Added cluster {cluster['cluster_id']} with {cluster['critical_count']} critical containers")
        
        # Early termination if we've covered all critical containers
        if len(critical_coverage) == len(critical_containers):
            logger.info("Stopping critical cluster selection: all critical containers covered")
            break
    
    logger.info(f"Selected {len(selected_clusters)} clusters containing critical containers")
    logger.info(f"Covered {len(critical_coverage)} out of {len(critical_containers)} critical containers")
    logger.info(f"Total containers in selected critical clusters: {total_containers}")
    
    return selected_clusters, critical_coverage

def handle_uncovered_critical_containers(
    uncovered_critical: Set[str],
    selected_clusters: Dict[str, List[str]],
    container_data: pd.DataFrame,
    all_clusters: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Handle critical containers not covered by initially selected clusters.
    
    Parameters
    ----------
    uncovered_critical : Set[str]
        Set of uncovered critical container IDs
    selected_clusters : Dict[str, List[str]]
        Currently selected clusters
    container_data : pd.DataFrame
        Container data with container details
    all_clusters : Dict[str, List[str]]
        All available clusters
        
    Returns
    -------
    Tuple[Dict[str, List[str]], Set[str]]
        - Updated selected clusters
        - Updated set of covered critical container IDs
    """
    updated_selected = selected_clusters.copy()
    critical_coverage = set()
    
    # Gather all containers already in selected clusters
    for containers in selected_clusters.values():
        critical_coverage.update(set(containers).intersection(uncovered_critical))
    
    remaining_uncovered = uncovered_critical - critical_coverage
    
    if remaining_uncovered:
        # Create container to cluster mapping for faster lookup
        container_to_cluster = {}
        for cluster_id, containers in all_clusters.items():
            for container_id in containers:
                container_to_cluster[container_id] = cluster_id
        
        # Process each uncovered critical container
        for container_id in remaining_uncovered:
           
            # Try to find the closest selected cluster
            closest_cluster_id = find_closest_cluster(
                container_id, updated_selected, container_data, all_clusters
            )
            
            ''' if container_id in container_to_cluster:
                # Container is in a cluster, add that cluster
                cluster_id = container_to_cluster[container_id]
                if cluster_id not in updated_selected:
                    updated_selected[cluster_id] = all_clusters[cluster_id]
                    critical_coverage.add(container_id)
            else:
                logger.warning(f"Critical container {container_id} not found in any cluster")
                '''
            
            if closest_cluster_id:
                # Add container to the closest cluster
                updated_selected[closest_cluster_id].append(container_id)
                critical_coverage.add(container_id)
            else:
                # Create a mini-cluster with this container
                new_cluster_id = f"critical_{container_id}"
                compatible_neighbors = find_compatible_neighbors(
                    container_id, container_data, 19, set(updated_selected.keys())
                )
                updated_selected[new_cluster_id] = [container_id] + compatible_neighbors
                critical_coverage.add(container_id)
    
    # Update critical_coverage with all critical containers
    for containers in updated_selected.values():
        critical_coverage.update(set(containers).intersection(uncovered_critical))
    
    return updated_selected, critical_coverage

def find_closest_cluster(
    container_id: str,
    selected_clusters: Dict[str, List[str]],
    container_data: pd.DataFrame,
    all_clusters: Dict[str, List[str]]
) -> Optional[str]:
    """
    Find the closest selected cluster for a container based on aisle proximity.
    
    Parameters
    ----------
    container_id : str
        Container ID to find closest cluster for
    selected_clusters : Dict[str, List[str]]
        Currently selected clusters
    container_data : pd.DataFrame
        Container data with container details
    all_clusters : Dict[str, List[str]]
        All available clusters
        
    Returns
    -------
    Optional[str]
        ID of the closest cluster, or None if no suitable cluster found
    """
    if not selected_clusters:
        return None
    
    # Get container aisle information
    container_aisles = get_container_aisles(container_id, container_data)
    if not container_aisles:
        logger.warning(f"No aisle information found for container {container_id}")
        return next(iter(selected_clusters))  # Return first cluster if no aisle info
    
    container_centroid = sum(container_aisles) / len(container_aisles)
    
    # Calculate centroid for each selected cluster
    closest_distance = float('inf')
    closest_cluster = None
    
    for cluster_id, containers in selected_clusters.items():
        cluster_aisles = []
        for cid in containers:
            cluster_aisles.extend(get_container_aisles(cid, container_data))
        
        if cluster_aisles:
            cluster_centroid = sum(cluster_aisles) / len(cluster_aisles)
            distance = abs(cluster_centroid - container_centroid)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_cluster = cluster_id
    
    return closest_cluster

def get_container_aisles(container_id: str, container_data: pd.DataFrame) -> List[int]:
    """
    Get aisle information for a container.
    
    Parameters
    ----------
    container_id : str
        Container ID
    container_data : pd.DataFrame
        Container data with container details
        
    Returns
    -------
    List[int]
        List of aisle numbers for the container
    """
    container_rows = container_data[container_data['container_id'] == container_id]
    if 'pick_aisle' in container_rows.columns:
        return container_rows['pick_aisle'].dropna().astype(int).tolist()
    return []

def find_compatible_neighbors(
    container_id: str,
    container_data: pd.DataFrame,
    k: int,
    exclude_clusters: Set[str]
) -> List[str]:
    """
    Find k-nearest compatible neighbors for a container based on aisle proximity.
    
    Parameters
    ----------
    container_id : str
        Container ID to find neighbors for
    container_data : pd.DataFrame
        Container data with container details
    k : int
        Number of neighbors to find
    exclude_clusters : Set[str]
        Set of cluster IDs to exclude containers from
        
    Returns
    -------
    List[str]
        List of compatible neighbor container IDs
    """
    container_aisles = get_container_aisles(container_id, container_data)
    if not container_aisles:
        return []
    
    container_centroid = sum(container_aisles) / len(container_aisles)
    
    # Calculate proximity to all other containers
    proximity = []
    for cid in container_data['container_id'].unique():
        if cid != container_id:
            aisles = get_container_aisles(cid, container_data)
            if aisles:
                centroid = sum(aisles) / len(aisles)
                distance = abs(centroid - container_centroid)
                proximity.append((cid, distance))
    
    # Sort by proximity and return k nearest
    proximity.sort(key=lambda x: x[1])
    return [cid for cid, _ in proximity[:k]]

def count_containers_in_clusters(clusters: Dict[str, List[str]]) -> int:
    """
    Count total number of containers in clusters.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
        
    Returns
    -------
    int
        Total container count
    """
    return sum(len(containers) for containers in clusters.values())

def score_clusters(
    clusters: Dict[str, List[str]],
    container_data: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Score clusters based on industry-standard metrics and aisle proximity.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_data : pd.DataFrame
        Container data with container details
        
    Returns
    -------
    List[Dict[str, Any]]
        List of scored clusters with their metadata
    """
    scored_clusters = []
    
    for cluster_id, containers in clusters.items():
        # Skip empty clusters
        if not containers:
            continue
        
        # Prepare data for scoring - extract aisle features
        cluster_features = []
        for container_id in containers:
            aisles = get_container_aisles(container_id, container_data)
            if aisles:
                centroid = sum(aisles) / len(aisles)
                span = max(aisles) - min(aisles) if len(aisles) > 1 else 0
                cluster_features.append([centroid, span])
        
        if not cluster_features:
            logger.warning(f"No aisle data for cluster {cluster_id}, cannot score")
            continue
            
        # Convert to numpy array for metric calculation
        features = np.array(cluster_features)
        
        # Only calculate metrics if we have enough samples
        if len(features) > 2:  # Need at least 3 samples for most metrics
            try:
                s_score = silhouette_score(features, np.zeros(len(features)))
            except:
                s_score = 0
                
            try:
                db_score = davies_bouldin_score(features, np.zeros(len(features)))
            except:
                db_score = float('inf')
                
            try:
                ch_score = calinski_harabasz_score(features, np.zeros(len(features)))
            except:
                ch_score = 0
        else:
            # Default scores for small clusters
            s_score = 0
            db_score = float('inf')
            ch_score = 0
        
        # Calculate aisle proximity score (inverse of average span)
        avg_span = np.mean([f[1] for f in cluster_features])
        aisle_proximity = 1 / (1 + avg_span)  # Normalized to (0,1]
        
        # Combined weighted score
        # These weights can be adjusted based on empirical testing
        w1, w2, w3, w4 = 0.3, 0.2, 0.2, 0.3
        
        # For Davies-Bouldin, lower is better, so we invert it
        db_component = 1 / (1 + db_score) if db_score != float('inf') else 0
        
        combined_score = (
            (w1 * s_score) + 
            (w2 * db_component) + 
            (w3 * (ch_score / 1000 if ch_score > 0 else 0)) +  # Normalize CH score
            (w4 * aisle_proximity)
        )
        
        scored_clusters.append({
            'cluster_id': cluster_id,
            'containers': containers,
            'size': len(containers),
            'score': combined_score,
            'metrics': {
                'silhouette': s_score,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'aisle_proximity': aisle_proximity
            }
        })
    
    # Sort by score (descending)
    scored_clusters.sort(key=lambda x: x['score'], reverse=True)
    return scored_clusters

def greedy_cluster_selection(
    scored_clusters: List[Dict[str, Any]],
    target_capacity: int
) -> Dict[str, List[str]]:
    """
    Select clusters using a greedy approach until target capacity is reached.
    
    Parameters
    ----------
    scored_clusters : List[Dict[str, Any]]
        List of scored clusters with their metadata
    target_capacity : int
        Target number of containers to select
        
    Returns
    -------
    Dict[str, List[str]]
        Selected clusters mapping cluster IDs to lists of container IDs
    """
    selected = {}
    current_count = 0
    
    for cluster in scored_clusters:
        selected[cluster['cluster_id']] = cluster['containers']
        current_count += cluster['size']
        
        if current_count >= target_capacity:
            break
    
    logger.info(f"Greedy selection added {len(selected)} clusters with {current_count} containers")
    return selected