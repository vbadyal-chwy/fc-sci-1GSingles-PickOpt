"""
Ultra-Simplified Linear Clustering Script

This script provides the most simplified approach to container clustering
with minimal functions and classes. Everything is done in a linear fashion
for maximum simplicity and ease of modification for other applications.

The script handles:
1. Feature extraction from container and slotbook data
2. Critical container identification and prioritization  
3. Hierarchical clustering with seed centers
4. Cluster augmentation and optimization
5. Tour calculation and final cluster assignment

All logic is contained in a single main function with minimal helper functions.
"""

import pandas as pd
import numpy as np
import logging
import time
import math
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#TODO: Changed max cluster size and containers per tour to volume centric values (cu in)
def linear_cluster_containers(container_data, slotbook_data, 
                             max_cluster_size=8000, max_volume_per_tour=2000, 
                             max_picking_capacity=1000, linkage_method='ward'):
    """
    Ultra-simplified linear clustering function.
    
    Parameters:
    -----------
    container_data : pd.DataFrame
        DataFrame with columns: container_id, item_number, slack_category (optional)
    slotbook_data : pd.DataFrame  
        DataFrame with columns: item_number, aisle_sequence
    max_cluster_size : int
        Maximum volume per cluster
    max_volume_per_tour : int
        Maximum volume per tour
    max_picking_capacity : int
        Maximum total containers to process
    linkage_method : str
        Hierarchical clustering linkage method
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to lists of container IDs
    """
    
    start_time = time.time()
    logger.info("Starting ultra-simplified linear clustering")
    
    # === STEP 1: DATA VALIDATION ===
    if container_data.empty or slotbook_data.empty:
        logger.error("Empty input data provided")
        return {}
    
    container_ids = container_data['container_id'].unique().tolist()
    logger.info(f"Processing {len(container_ids)} unique containers")

    # Create a mapping from container_id to its volume
    container_vol_map = dict(zip(container_data['container_id'], container_data['volume']))
    
    container_vol = [container_vol_map[c_id] for c_id in container_ids]
    total_volume = sum(container_vol)
    logger.info(f"Total volume: {total_volume}")
    
    # Early return for small datasets
    if total_volume <= max_cluster_size:
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
        # === STANDARD CLUSTERING PATH ===
        logger.info("Using standard clustering path")

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
        
        volume_valid_containers = [container_vol_map[c_id] for c_id in valid_containers]
        total_volume_valid_containers = sum(volume_valid_containers)
        logger.info(f"Total volume of valid containers: {total_volume_valid_containers}")

        # Determine optimal number of clusters
        estimated_clusters = max(2, total_volume_valid_containers // max_cluster_size + 1)
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
            # calculate total vol for all containers in this cluster
            cluster_volume = sum(container_vol_map.get(c_id, 0) for c_id in containers)
            logger.info(f"Cluster {cluster_id} contains {len(containers)} containers with total volume {cluster_volume}")

            if cluster_volume <= max_cluster_size:
                final_clusters[str(cluster_counter)] = containers
                cluster_counter += 1
            else:
                # Subdivide oversized cluster recursively
                sub_clusters = _subdivide_cluster_simple(containers, container_features, container_vol_map, max_cluster_size, linkage_method)
                for sub_containers in sub_clusters.values():
                    final_clusters[str(cluster_counter)] = sub_containers
                    cluster_counter += 1
        
    else:
        # === CRITICAL CONTAINER PRIORITIZATION PATH ===
        logger.info("Using critical container prioritization path")
        
        # Separate containers
        non_critical_containers = [c_id for c_id in container_ids if c_id not in critical_containers]
        
        # If critical containers exceed capacity, prioritize them only
        if len(critical_containers) > max_picking_capacity:
            non_critical_containers = []
        
        # === STEP 1: Form seed clusters from critical containers ===
        # Prepare feature matrix for critical containers
        feature_arrays = []
        valid_critical_containers = []
        
        for c_id in critical_containers:
            if c_id in container_features:
                centroid, span, _ = container_features[c_id]
                feature_arrays.append([centroid, span])
                valid_critical_containers.append(c_id)
        
        if len(valid_critical_containers) < 2:
            if len(valid_critical_containers) == 1:
                seed_clusters = {0: valid_critical_containers}
            else:
                logger.error("No valid critical containers found")
                return {}
        else:
            # Normalize features
            feature_matrix = np.array(feature_arrays)
            feature_matrix = (feature_matrix - feature_matrix.min(axis=0)) / (feature_matrix.max(axis=0) - feature_matrix.min(axis=0) + 1e-8)
            
            # Determine optimal number of clusters
            
            # Calculate total volume for valid critical containers
            cluster_volume = sum(container_vol_map.get(c_id, 0) for c_id in valid_critical_containers)
            logger.info(f"Total volume for valid critical containers: {cluster_volume}")

            min_clusters = max(2, cluster_volume // max_cluster_size + 1)
            max_clusters = min(10, len(valid_critical_containers) // 2)
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
            for i, container_id in enumerate(valid_critical_containers):
                cluster_id = int(best_clusters[i]) - 1  # Convert to 0-based
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(container_id)
            
            # Break down oversized clusters
            seed_clusters = {}
            cluster_counter = 0            
            
            for cluster_id, containers in clusters.items():
                cluster_volume = sum(container_vol_map.get(c_id, 0) for c_id in containers)
                logger.info(f"Cluster {cluster_id} contains {len(containers)} containers with total volume {cluster_volume}")

                if cluster_volume <= max_cluster_size:
                    seed_clusters[cluster_counter] = containers
                    cluster_counter += 1
                else:
                    # Subdivide oversized cluster
                    sub_clusters = _subdivide_cluster_simple(containers, container_features, container_vol_map, max_cluster_size, linkage_method)
                    for sub_containers in sub_clusters.values():
                        seed_clusters[cluster_counter] = sub_containers
                        cluster_counter += 1
        
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
                
                #TODO: bug here if closest_cluster is 0 it will not be added
                if closest_cluster is not None:
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
        critical_set = set(critical_containers)
        final_clusters = {k: v.copy() for k, v in merged_clusters.items()}
        remaining = set(remaining_non_critical)
        
        # For each cluster, add non-critical containers up to capacity
        for cluster_id, containers in final_clusters.items():
            # Count critical containers
            critical_volume = sum(container_vol_map.get(c_id, 0) for c_id in containers if c_id in critical_set)
            space_left = max_cluster_size - critical_volume
            
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
                
                containers_to_add = []
                total_added_volume = 0
                for c_id, _ in container_distances:
                    c_vol = container_vol_map.get(c_id, 0)
                    if total_added_volume + c_vol <= space_left:
                        containers_to_add.append(c_id)
                        total_added_volume += c_vol
                    else:
                        break
                
                final_clusters[cluster_id].extend(containers_to_add)
                
                # Remove from remaining pool
                for c_id in containers_to_add:
                    if c_id in remaining:
                        remaining.remove(c_id)
        
        remaining_containers = list(remaining)
        
        #TODO: Update here to use volume centric
        # === STEP 5: Handle remaining capacity with additional clusters ===
        total_seed_containers = sum([len(c) for c in final_clusters.values()])
        remaining_capacity = max_picking_capacity - total_seed_containers
        
        if remaining_capacity > 0 and remaining_containers:
            # Form additional clusters from remaining containers using standard path
            # Prepare feature matrix for remaining containers
            feature_arrays = []
            valid_remaining = []
            
            for c_id in remaining_containers:
                if c_id in container_features:
                    centroid, span, _ = container_features[c_id]
                    feature_arrays.append([centroid, span])
                    valid_remaining.append(c_id)
            
            if valid_remaining:
                # Normalize features
                feature_matrix = np.array(feature_arrays)
                feature_matrix = (feature_matrix - feature_matrix.min(axis=0)) / (feature_matrix.max(axis=0) - feature_matrix.min(axis=0) + 1e-8)
                
                valid_container_volume = sum(container_vol_map.get(c_id, 0) for c_id in valid_remaining)
                # Determine optimal number of clusters
                estimated_clusters = max(2, valid_container_volume // max_cluster_size + 1)
                max_possible = min(10, len(valid_remaining) // 2)
                optimal_clusters = min(estimated_clusters, max_possible)
                
                # Perform hierarchical clustering
                Z = linkage(feature_matrix, method=linkage_method)
                cluster_labels = fcluster(Z, optimal_clusters, criterion='maxclust')
                
                # Map containers to clusters
                additional_clusters = {}
                for i, container_id in enumerate(valid_remaining):
                    cluster_id = int(cluster_labels[i])
                    if cluster_id not in additional_clusters:
                        additional_clusters[cluster_id] = []
                    additional_clusters[cluster_id].append(container_id)
                
                # Break down oversized clusters
                processed_additional = {}
                cluster_counter = 1
                
                for cluster_id, containers in additional_clusters.items():
                    cluster_volume = sum(container_vol_map.get(c_id, 0) for c_id in containers)
                    logger.info(f"Cluster {cluster_id} contains {len(containers)} containers with total volume {cluster_volume}")

                    if cluster_volume <= max_cluster_size:
                        processed_additional[str(cluster_counter)] = containers
                        cluster_counter += 1
                    else:
                        # Subdivide oversized cluster
                        print(f"Subdividing oversized cluster {cluster_id} with volume {cluster_volume}")
                        sub_clusters = _subdivide_cluster_simple(containers, container_features, container_vol_map, max_cluster_size, linkage_method)
                        for sub_containers in sub_clusters.values():
                            processed_additional[str(cluster_counter)] = sub_containers
                            cluster_counter += 1
                
                # Select best clusters to fill remaining capacity
                cluster_metrics = {}
                for cluster_id, containers in processed_additional.items():
                    cluster_volume = sum(container_vol_map.get(c_id, 0) for c_id in containers)
                    # Simple quality metric based on spatial cohesion
                    centroids = [container_features[c_id][0] for c_id in containers if c_id in container_features]
                    spans = [container_features[c_id][1] for c_id in containers if c_id in container_features]
                    
                    if centroids:
                        centroid_std = np.std(centroids) if len(centroids) > 1 else 0
                        span_std = np.std(spans) if len(spans) > 1 else 0
                        quality = 1.0 / (centroid_std + span_std + 0.1)  # Higher is better
                    else:
                        quality = 0
                    
                    #TODO: Volume centric
                    tours = max(1, cluster_volume // max_volume_per_tour + 1)
                    
                    cluster_metrics[cluster_id] = {
                        'containers': containers,
                        'quality': quality,
                        'tours': tours
                    }
                
                # Sort by quality and select greedily
                sorted_clusters = sorted(cluster_metrics.items(), key=lambda x: x[1]['quality'], reverse=True)
                
                selected_additional = {}
                total_containers = 0
                
                for cluster_id, info in sorted_clusters:
                    selected_additional[cluster_id] = info['containers']
                    total_containers += len(info['containers'])
                    
                    if total_containers >= remaining_capacity:
                        break
                
                # Add selected clusters to final result
                next_id = len(final_clusters) + 1
                for containers in selected_additional.values():
                    final_clusters[str(next_id)] = containers
                    next_id += 1
    
    # === STEP 6: FINALIZE AND RENUMBER CLUSTERS ===
    # Sort clusters by minimum aisle for logical ordering
    sorted_clusters = []
    for cluster_id, containers in final_clusters.items():
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
    
    # === STEP 7: LOG RESULTS ===
    total_time = time.time() - start_time
    #TODO: Volume centric
    total_containers = sum(len(c) for c in final_clusters.values())
    total_tours = sum(math.ceil(sum(container_vol_map.get(c_id, 0) for c_id in c) / max_volume_per_tour) for c in final_clusters.values())
    
    logger.info(f"Clustering completed in {total_time:.2f} seconds")
    logger.info(f"Formed {len(final_clusters)} clusters with {total_containers} containers")
    logger.info(f"Total tours required: {total_tours}")
    
    # Print cluster statistics
    for cluster_id, containers in final_clusters.items():
        critical_count = sum(1 for c in containers if c in critical_containers)
        tours = math.ceil(sum(container_vol_map.get(c_id, 0) for c_id in containers) / max_volume_per_tour)
        logger.info(f"Cluster {cluster_id}: {len(containers)} containers ({critical_count} critical) - {tours} tours")
    
    return final_clusters


def _subdivide_cluster_simple(containers, container_features, container_vol_map, max_cluster_size, linkage_method, max_depth=3, depth=0):
    """Simple recursive function to subdivide oversized clusters."""
    
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
        sub_volume = sum(container_vol_map.get(c_id, 0) for c_id in sub_containers)
        if sub_volume <= max_cluster_size:
            final_sub_clusters[cluster_counter] = sub_containers
            cluster_counter += 1
        else:
            # Recursively subdivide
            recursive_clusters = _subdivide_cluster_simple(
                sub_containers, container_features, container_vol_map, max_cluster_size, 
                linkage_method, max_depth, depth + 1
            )
            for recursive_containers in recursive_clusters.values():
                final_sub_clusters[cluster_counter] = recursive_containers
                cluster_counter += 1
    
    return final_sub_clusters


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Example usage with sample data
    logger.info("Running example linear clustering")
    
    # Create sample data
    np.random.seed(42)
    
    # Create a set of 25 random SKUs
    sku_list = [f'SKU{i:03d}' for i in range(1, 200)]
    num_containers = 10000
    # Sample container data with random SKU assignment
    container_data = pd.DataFrame({
        'container_id': [f'C{i:03d}' for i in range(1, num_containers + 1)],
        'item_number': np.random.choice(sku_list, num_containers),
        'slack_category': np.random.choice(['Normal', 'Critical', 'Urgent'], num_containers, p=[0.95, 0.025, 0.025]),
        'volume': np.random.randint(150, 2000, num_containers)
    })
    
    # Sample slotbook data
    slotbook_data = pd.DataFrame({
        'item_number': sku_list,
        'aisle_sequence': np.random.randint(1, 91, len(sku_list))
    })
    
    # Run clustering
    clusters = linear_cluster_containers(
        container_data=container_data,
        slotbook_data=slotbook_data,
        max_cluster_size=66000*2,
        max_volume_per_tour=66000,
        max_picking_capacity=10*120
    )
    
    logger.info("Example linear clustering completed")
    # Save container and slotbook data to CSV
    container_data.to_csv("clustering_container_data.csv", index=False)
    slotbook_data.to_csv("clustering_slotbook_data.csv", index=False)

    # Save cluster details to CSV
    cluster_rows = []
    # Build a mapping from container_id to slack_category for quick lookup
    container_slack_map = dict(zip(container_data['container_id'], container_data['slack_category']))
    for cluster_id, containers in clusters.items():
        for c_id in containers:
            slack_category = container_slack_map.get(c_id, "Normal")
            is_critical = int(slack_category in ["Critical", "Urgent"])
            sku = container_data.loc[container_data['container_id'] == c_id, 'item_number'].values[0]
            # Get the aisle_sequence for the SKU from slotbook_data
            sku_location = slotbook_data.loc[slotbook_data['item_number'] == sku, 'aisle_sequence'].values
            sku_location = sku_location[0] if len(sku_location) > 0 else None
            cluster_rows.append({
                "cluster_id": cluster_id,
                "container_id": c_id,
                "slack_category": slack_category,
                "is_critical": is_critical,
                "sku": sku,
                "sku_location": sku_location
            })
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv("cluster_details.csv", index=False)