import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
import time
import math
from collections import defaultdict, Counter
import warnings
import random
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

def create_sku_aisle_mapping(slotbook_data: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Create a mapping from SKUs to their aisle locations.
    
    Parameters
    ----------
    slotbook_data : pd.DataFrame
        DataFrame containing SKU inventory information with columns 'item_number' and 'aisle_sequence'
        
    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping SKU IDs to lists of aisle numbers
    """
    sku_aisle_mapping = {}
    
    # Process each row in the slotbook data
    for _, row in slotbook_data.iterrows():
        sku = row['item_number']
        aisle = row['aisle_sequence']
        
        if sku not in sku_aisle_mapping:
            sku_aisle_mapping[sku] = []
        
        # Only add unique aisles
        if aisle not in sku_aisle_mapping[sku]:
            sku_aisle_mapping[sku].append(aisle)
    
    # Sort aisle lists for deterministic behavior
    for sku in sku_aisle_mapping:
        sku_aisle_mapping[sku].sort()
    
    return sku_aisle_mapping

def get_container_aisles(container_id: str, container_data: pd.DataFrame, 
                        sku_aisle_mapping: Dict[str, List[int]]) -> Set[int]:
    """
    Get optimized aisles required for a specific container, minimizing total aisles visited.
    
    Parameters
    ----------
    container_id : str
        ID of the container
    container_data : pd.DataFrame
        Container data with SKU and quantity information
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
        
    Returns
    -------
    Set[int]
        Set of aisle numbers required for this container
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
            # If one or more aisles are already covered, pick the first one
            best_aisle = already_covered[0]
        else:
            # Otherwise, find the aisle that minimizes the distance to the nearest must-visit aisle
            if not must_visit_aisles:
                best_aisle = aisles[0]
            else:
                # Calculate "distance" to the nearest must-visit aisle for each option
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

def extract_container_features(container_id: str, container_data: pd.DataFrame, 
                              sku_aisle_mapping: Dict[str, List[int]]) -> Dict[str, float]:
    """
    Extract numerical features for a container to be used in the graph.
    
    Parameters
    ----------
    container_id : str
        ID of the container
    container_data : pd.DataFrame
        Container data with order details
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
        
    Returns
    -------
    Dict[str, float]
        Dictionary of container features
    """
    # Get container's rows from the dataframe
    container_rows = container_data[container_data['container_id'] == container_id]
    
    # Skip if no data found
    if container_rows.empty:
        return {
            'aisle_centroid': 0.0,
            'aisle_span': 0.0,
            'distinct_aisles': 0.0,
            'sku_count': 0.0,
            'total_quantity': 0.0,
            'is_critical': 0.0,
            'slack_value': 0.0
        }
    
    # Calculate basic order metrics
    sku_count = len(container_rows['item_number'].unique())
    total_quantity = container_rows['quantity'].sum()
    
    # Get container aisles
    aisles = get_container_aisles(container_id, container_data, sku_aisle_mapping)
    
    # Calculate aisle metrics
    if aisles:
        aisle_centroid = sum(aisles) / len(aisles)
        aisle_min = min(aisles)
        aisle_max = max(aisles)
        aisle_span = aisle_max - aisle_min if len(aisles) > 1 else 0
        distinct_aisles = len(aisles)
    else:
        aisle_centroid = 0
        aisle_span = 0
        distinct_aisles = 0
    
    # Check if container is critical (has slack data)
    is_critical = 0.0
    slack_value = 0.0
    
    if 'slack_category' in container_rows.columns:
        # Use most common category if multiple rows
        slack_category = container_rows['slack_category'].mode().iloc[0]
        is_critical = 1.0 if slack_category == 'Critical' else 0.0
        
        # Convert slack categories to numerical values
        if slack_category == 'Critical':
            slack_value = 10.0
        elif slack_category == 'Urgent':
            slack_value = 5.0
        else:  # 'Safe'
            slack_value = 1.0
    
    # If slack_minutes is available, use that directly
    if 'slack_minutes' in container_rows.columns:
        slack_minutes = container_rows['slack_minutes'].mean()
        # Normalize slack minutes to a reasonable range
        slack_value = max(10.0, min(1.0, 10.0 * (1 / (1 + max(0, slack_minutes / 60)))))
    
    return {
        'aisle_centroid': float(aisle_centroid),
        'aisle_span': float(aisle_span),
        'distinct_aisles': float(distinct_aisles),
        'sku_count': float(sku_count),
        'total_quantity': float(total_quantity),
        'is_critical': float(is_critical),
        'slack_value': float(slack_value)
    }

def build_container_graph(container_data: pd.DataFrame, slotbook_data: pd.DataFrame,
                         container_ids: Optional[List[str]] = None) -> nx.Graph:
    """
    Build a graph representation of container relationships.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory information
    container_ids : Optional[List[str]]
        Optional list of specific container IDs to include
        
    Returns
    -------
    nx.Graph
        NetworkX graph with containers as nodes and edges representing picking affinity
    """
    start_time = time.time()
    logger.info("Building container graph...")
    
    # Create SKU to aisle mapping
    sku_aisle_mapping = create_sku_aisle_mapping(slotbook_data)
    
    # Initialize graph
    G = nx.Graph()
    
    # Filter container IDs if specified
    if container_ids is not None:
        selected_container_ids = container_ids
    else:
        selected_container_ids = container_data['container_id'].unique()
    
    # Limit to a reasonable number if needed
    MAX_CONTAINERS = 15000  # Safety limit
    if len(selected_container_ids) > MAX_CONTAINERS:
        logger.warning(f"Too many containers ({len(selected_container_ids)}), limiting to {MAX_CONTAINERS}")
        selected_container_ids = selected_container_ids[:MAX_CONTAINERS]
    
    # Add nodes with features
    for container_id in selected_container_ids:
        node_features = extract_container_features(container_id, container_data, sku_aisle_mapping)
        G.add_node(container_id, features=node_features)
    
    # Avoid O(n²) edge creation by using spatial partitioning
    # Group containers by approximate aisle regions
    aisle_partitions = defaultdict(list)
    
    for container_id in G.nodes():
        aisles = get_container_aisles(container_id, container_data, sku_aisle_mapping)
        if aisles:
            aisle_min, aisle_max = min(aisles), max(aisles)
            # Create partition key based on aisle range
            partition_key = (aisle_min // 10, aisle_max // 10)  # Group by aisle ranges of size 10
            aisle_partitions[partition_key].append(container_id)
    
    # Add edges between containers in same or adjacent partitions
    edges_added = 0
    
    # Create a fixed list of partition items to avoid modification during iteration
    partition_items = list(aisle_partitions.items())
    
    for partition_key, partition_containers in partition_items:
        # Find adjacent partitions 
        min_group, max_group = partition_key
        adjacent_keys = [
            (min_group-1, max_group), (min_group+1, max_group),
            (min_group, max_group-1), (min_group, max_group+1),
            (min_group-1, max_group-1), (min_group+1, max_group+1),
            (min_group-1, max_group+1), (min_group+1, max_group-1)
        ]
        
        # Get all containers in this partition and adjacent partitions
        all_relevant_containers = partition_containers.copy()
        for adj_key in adjacent_keys:
            all_relevant_containers.extend(aisle_partitions[adj_key])
        
        # Create edges within this extended group
        for i, container_i in enumerate(partition_containers):
            aisles_i = get_container_aisles(container_i, container_data, sku_aisle_mapping)
            features_i = G.nodes[container_i]['features']
            
            for container_j in all_relevant_containers[partition_containers.index(container_i)+1 if container_i in partition_containers else 0:]:
                if container_i == container_j:
                    continue
                    
                aisles_j = get_container_aisles(container_j, container_data, sku_aisle_mapping)
                features_j = G.nodes[container_j]['features']
                
                # Calculate affinity metrics
                if aisles_i and aisles_j:
                    # Calculate aisle overlap
                    aisle_intersection = set(aisles_i).intersection(set(aisles_j))
                    aisle_union = set(aisles_i).union(set(aisles_j))
                    overlap_count = len(aisle_intersection)
                    overlap_ratio = overlap_count / len(aisle_union) if aisle_union else 0
                    
                    # Calculate centroid distance
                    centroid_distance = abs(features_i['aisle_centroid'] - features_j['aisle_centroid'])
                    
                    # Calculate combined span effect
                    combined_min = min(min(aisles_i), min(aisles_j))
                    combined_max = max(max(aisles_i), max(aisles_j))
                    combined_span = combined_max - combined_min
                    
                    # Current individual spans
                    max_individual_span = max(features_i['aisle_span'], features_j['aisle_span'])
                    span_increase = combined_span - max_individual_span
                    
                    # Only add edge if there's meaningful affinity
                    if overlap_count > 0 or centroid_distance < 10:
                        edge_features = {
                            'overlap_count': float(overlap_count),
                            'overlap_ratio': float(overlap_ratio),
                            'centroid_distance': float(centroid_distance),
                            'combined_span': float(combined_span),
                            'span_increase': float(span_increase)
                        }
                        
                        G.add_edge(container_i, container_j, features=edge_features)
                        edges_added += 1
    
    logger.info(f"Built container graph with {len(G.nodes())} nodes and {edges_added} edges in {time.time() - start_time:.2f} seconds")
    return G

class ContainerGNN(nn.Module):
    """
    Graph Neural Network for container embeddings.
    
    This GNN model processes container nodes and their relationships to create embeddings
    that capture picking affinity and can be used for clustering.
    """
    
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64, embedding_dim: int = 32):
        """
        Initialize the GNN model.
        
        Parameters
        ----------
        node_feature_dim : int
            Number of features per container node
        edge_feature_dim : int
            Number of features per container edge
        hidden_dim : int
            Dimension of hidden layers
        embedding_dim : int
            Dimension of final embeddings
        """
        super(ContainerGNN, self).__init__()
        
        # Feature transformation layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
        # Normalization after convolutions
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the GNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, node_feature_dim]
        edge_index : torch.Tensor
            Graph connectivity in COO format [2, num_edges]
        edge_attr : torch.Tensor, optional
            Edge features [num_edges, edge_feature_dim]
            
        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, embedding_dim]
        """
        # Encode node features
        x = self.node_encoder(x)
        
        # Apply graph convolutions with skip connections
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(self.bn1(x1))
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(self.bn2(x2))
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x2, edge_index)
        embeddings = self.bn3(x3)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def convert_to_pytorch_geometric(G: nx.Graph) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric data object.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph with node and edge features
        
    Returns
    -------
    Data
        PyTorch Geometric data object
    """
    # Get container IDs for indexing
    container_ids = list(G.nodes())
    container_id_to_idx = {container_id: i for i, container_id in enumerate(container_ids)}
    
    # Prepare node features tensor
    node_features = []
    for container_id in container_ids:
        features = G.nodes[container_id]['features']
        # Convert dictionary to list in a consistent order
        feature_list = [
            features.get('aisle_centroid', 0.0),
            features.get('aisle_span', 0.0),
            features.get('distinct_aisles', 0.0),
            features.get('sku_count', 0.0),
            features.get('total_quantity', 0.0),
            features.get('is_critical', 0.0),
            features.get('slack_value', 1.0)
        ]
        node_features.append(feature_list)
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Prepare edge indices and attributes
    edge_indices = []
    edge_attrs = []
    
    for source, target, data in G.edges(data=True):
        # Get node indices
        source_idx = container_id_to_idx[source]
        target_idx = container_id_to_idx[target]
        
        # Add edge in both directions (undirected graph)
        edge_indices.append([source_idx, target_idx])
        edge_indices.append([target_idx, source_idx])
        
        # Add edge features
        if 'features' in data:
            features = data['features']
            feature_list = [
                features.get('overlap_count', 0.0),
                features.get('overlap_ratio', 0.0),
                features.get('centroid_distance', 0.0),
                features.get('combined_span', 0.0),
                features.get('span_increase', 0.0)
            ]
            # Add features for both directions
            edge_attrs.append(feature_list)
            edge_attrs.append(feature_list)
    
    # Convert to tensors if we have edges
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Create empty tensors if no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Store container IDs for later reference
    data.container_ids = container_ids
    
    return data

def get_critical_containers(container_data: pd.DataFrame) -> Set[str]:
    """
    Extract IDs of critical containers from container data.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with slack information
        
    Returns
    -------
    Set[str]
        Set of critical container IDs
    """
    critical_containers = set()
    
    # Check if slack information is available
    if 'slack_category' in container_data.columns:
        # Get unique container IDs with 'Critical' slack category
        critical_df = container_data[container_data['slack_category'] == 'Critical']
        critical_containers = set(critical_df['container_id'].unique())
        
    # Additional logic for slack_minutes if available
    elif 'slack_minutes' in container_data.columns:
        # Consider containers with negative slack as critical
        critical_df = container_data[container_data['slack_minutes'] < 0]
        critical_containers = set(critical_df['container_id'].unique())
    
    logger.info(f"Identified {len(critical_containers)} critical containers")
    return critical_containers

def generate_training_pairs(container_graph: nx.Graph, container_data: pd.DataFrame, 
                           sku_aisle_mapping: Dict[str, List[int]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate positive and negative training pairs for contrastive learning.
    
    Parameters
    ----------
    container_graph : nx.Graph
        Container graph with node and edge features
    container_data : pd.DataFrame
        Container data with order details
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
        
    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        Lists of positive and negative container pairs
    """
    logger.info("Generating training pairs for GNN...")
    positive_pairs = []
    negative_pairs = []
    
    # Extract container features for reference
    container_features = {}
    for container_id in container_graph.nodes():
        aisles = get_container_aisles(container_id, container_data, sku_aisle_mapping)
        if not aisles:
            continue
            
        container_features[container_id] = {
            'aisles': aisles,
            'aisle_min': min(aisles),
            'aisle_max': max(aisles),
            'aisle_span': max(aisles) - min(aisles) if len(aisles) > 1 else 0,
            'centroid': sum(aisles)/len(aisles)
        }
    
    # Examine each edge in the graph
    for container_i, container_j, data in container_graph.edges(data=True):
        # Skip if either container doesn't have features
        if container_i not in container_features or container_j not in container_features:
            continue
            
        # Get features for both containers
        feat_i = container_features[container_i]
        feat_j = container_features[container_j]
        
        # Calculate combined metrics
        combined_min = min(feat_i['aisle_min'], feat_j['aisle_min'])
        combined_max = max(feat_i['aisle_max'], feat_j['aisle_max'])
        combined_span = combined_max - combined_min
        span_increase = combined_span - max(feat_i['aisle_span'], feat_j['aisle_span'])
        
        # Get edge features
        edge_features = data.get('features', {})
        overlap_ratio = edge_features.get('overlap_ratio', 0.0)
        
        # POSITIVE PAIR CRITERIA
        
        # Criterion 1: High aisle overlap with minimal span increase
        if overlap_ratio > 0.5 and span_increase < 3:
            positive_pairs.append((container_i, container_j))
            continue
            
        '''# Criterion 2: Adjacent aisles (even without overlap)
        distance_between_spans = max(0, 
                               min(feat_i['aisle_max'], feat_j['aisle_max']) - 
                               max(feat_i['aisle_min'], feat_j['aisle_min']))
        if distance_between_spans <= 2 and combined_span < 10:
            positive_pairs.append((container_i, container_j))
            continue'''
            
        # Criterion 3: Contained spans (one completely inside another)
        if ((feat_i['aisle_min'] >= feat_j['aisle_min'] and feat_i['aisle_max'] <= feat_j['aisle_max']) or \
           (feat_j['aisle_min'] >= feat_i['aisle_min'] and feat_j['aisle_max'] <= feat_i['aisle_max'])) \
            and span_increase < 3:
            positive_pairs.append((container_i, container_j))
            continue
        
        # NEGATIVE PAIR CRITERIA
        
        # Criterion 1: No overlap and distant centroids
        if overlap_ratio == 0 and abs(feat_i['centroid'] - feat_j['centroid']) > 3:
            negative_pairs.append((container_i, container_j))
            continue
            
        # Criterion 2: Extreme span increase
        if span_increase > 3:
            negative_pairs.append((container_i, container_j))
            continue
            
        # Criterion 3: Distant non-overlapping aisle ranges
        gap_between_spans = max(0, 
                            min(feat_j['aisle_min'], feat_i['aisle_min']) - 
                            max(feat_j['aisle_max'], feat_i['aisle_max']))
        if gap_between_spans > 3:
            negative_pairs.append((container_i, container_j))
            continue
    
    # Add transitive positive relationships
    # If A->B and B->C are positive, then A->C is likely positive
    positive_pairs_set = set(positive_pairs)
    container_to_positive = defaultdict(set)
    
    for c1, c2 in positive_pairs:
        container_to_positive[c1].add(c2)
        container_to_positive[c2].add(c1)
    
    transitive_pairs = []
    for c1, connected in container_to_positive.items():
        if len(connected) < 2:
            continue
            
        connected_list = list(connected)
        for i in range(len(connected_list)):
            for j in range(i+1, len(connected_list)):
                c2, c3 = connected_list[i], connected_list[j]
                
                # Check if not already a direct positive pair
                if (c2, c3) not in positive_pairs_set and (c3, c2) not in positive_pairs_set:
                    transitive_pairs.append((c2, c3))
    
    # Add a subset of transitive pairs
    if transitive_pairs:
        sample_size = min(len(transitive_pairs), len(positive_pairs) // 2)
        sampled_transitive = random.sample(transitive_pairs, sample_size)
        positive_pairs.extend(sampled_transitive)
    
    # Ensure no overlap between positive and negative pairs
    positive_pairs_set = set(tuple(sorted(pair)) for pair in positive_pairs)
    negative_pairs_filtered = [pair for pair in negative_pairs 
                             if tuple(sorted(pair)) not in positive_pairs_set]
    
    # Balance the datasets
    max_pairs = 5000  # Limit for very large graphs
    
    if len(positive_pairs) > max_pairs:
        positive_pairs = random.sample(positive_pairs, max_pairs)
        
    if len(negative_pairs_filtered) > max_pairs:
        negative_pairs_filtered = random.sample(negative_pairs_filtered, max_pairs)
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs_filtered)} negative pairs")
    return positive_pairs, negative_pairs_filtered

def train_gnn(container_graph: nx.Graph, container_data: pd.DataFrame, 
             sku_aisle_mapping: Dict[str, List[int]], epochs: int = 50, 
             learning_rate: float = 0.001, device: str = None) -> ContainerGNN:
    """
    Train the GNN model using contrastive learning.
    
    Parameters
    ----------
    container_graph : nx.Graph
        Container graph with node and edge features
    container_data : pd.DataFrame
        Container data with order details
    sku_aisle_mapping : Dict[str, List[int]]
        Mapping of SKUs to their aisle locations
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    device : str
        Device to use for training ('cuda' or 'cpu')
        
    Returns
    -------
    ContainerGNN
        Trained GNN model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Training GNN on {device}...")
    
    # Convert graph to PyTorch Geometric format
    data = convert_to_pytorch_geometric(container_graph)
    data = data.to(device)
    
    # Generate training pairs
    positive_pairs, negative_pairs = generate_training_pairs(
        container_graph, container_data, sku_aisle_mapping
    )
    
    # Get container ID to index mapping
    container_ids = data.container_ids
    container_id_to_idx = {container_id: i for i, container_id in enumerate(container_ids)}
    
    # Convert pairs to tensor indices
    pos_pairs_idx = torch.tensor([
        [container_id_to_idx[p[0]], container_id_to_idx[p[1]]]
        for p in positive_pairs
        if p[0] in container_id_to_idx and p[1] in container_id_to_idx
    ], device=device)
    
    neg_pairs_idx = torch.tensor([
        [container_id_to_idx[p[0]], container_id_to_idx[p[1]]]
        for p in negative_pairs
        if p[0] in container_id_to_idx and p[1] in container_id_to_idx
    ], device=device)
    
    # Initialize model
    node_feature_dim = data.x.shape[1]
    edge_feature_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0
    
    model = ContainerGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=64,
        embedding_dim=32
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    model.train()
    
    # Progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        # Contrastive loss
        loss = 0.0
        margin = 1.0  # Margin for triplet loss
        
        # Positive pairs - minimize distance
        if len(pos_pairs_idx) > 0:
            anchor_embeds = embeddings[pos_pairs_idx[:, 0]]
            positive_embeds = embeddings[pos_pairs_idx[:, 1]]
            pos_distance = F.pairwise_distance(anchor_embeds, positive_embeds)
            positive_loss = pos_distance.mean()
            loss += positive_loss
        
        # Negative pairs - maximize distance
        if len(neg_pairs_idx) > 0:
            anchor_embeds = embeddings[neg_pairs_idx[:, 0]]
            negative_embeds = embeddings[neg_pairs_idx[:, 1]]
            neg_distance = F.pairwise_distance(anchor_embeds, negative_embeds)
            negative_loss = torch.clamp(margin - neg_distance, min=0).mean()
            loss += negative_loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Learning rate scheduler
        scheduler.step(loss)
    
    logger.info(f"GNN training completed with final loss: {loss.item():.4f}")
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def generate_embeddings(gnn_model: ContainerGNN, container_graph: nx.Graph, device: str = None) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for all containers using the trained GNN model.
    
    Parameters
    ----------
    gnn_model : ContainerGNN
        Trained GNN model
    container_graph : nx.Graph
        Container graph with node and edge features
    device : str
        Device to use for inference
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping container IDs to embedding vectors
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Generating container embeddings on {device}...")
    
    # Convert graph to PyTorch Geometric format
    data = convert_to_pytorch_geometric(container_graph)
    data = data.to(device)
    
    # Ensure model is in evaluation mode
    gnn_model.eval()
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = gnn_model(data.x, data.edge_index, data.edge_attr)
    
    # Convert to numpy and create mapping to container IDs
    embeddings_np = embeddings.cpu().numpy()
    container_embeddings = {
        container_id: embeddings_np[i]
        for i, container_id in enumerate(data.container_ids)
    }
    
    logger.info(f"Generated embeddings for {len(container_embeddings)} containers")
    return container_embeddings

def cluster_with_constraints(container_embeddings: Dict[str, np.ndarray], 
                            container_data: pd.DataFrame,
                            max_cluster_size: int = 200,
                            containers_per_tour: int = 20,
                            clustering_method: str = 'kmeans') -> Dict[str, List[str]]:
    """
    Cluster container embeddings while respecting max cluster size constraint.
    
    Parameters
    ----------
    container_embeddings : Dict[str, np.ndarray]
        Dictionary mapping container IDs to embedding vectors
    container_data : pd.DataFrame
        Container data with order details
    max_cluster_size : int
        Maximum allowed cluster size
    containers_per_tour : int
        Number of containers per tour
    clustering_method : str
        Clustering method to use ('kmeans' or 'dbscan')
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    logger.info(f"Clustering container embeddings with {clustering_method}...")
    
    # Extract critical containers
    critical_containers = get_critical_containers(container_data)
    
    # Prepare embedding matrix and container IDs
    container_ids = list(container_embeddings.keys())
    embedding_matrix = np.array([container_embeddings[c_id] for c_id in container_ids])
    
    # Get count of critical containers in our embedding set
    critical_in_scope = [c_id for c_id in container_ids if c_id in critical_containers]
    critical_count = len(critical_in_scope)
    
    logger.info(f"Clustering {len(container_ids)} containers with {critical_count} critical containers")
    
    # Calculate number of clusters to create
    total_containers = len(container_ids)
    
    # Calculate needed tours - each tour can handle containers_per_tour containers
    needed_tours = math.ceil(total_containers / containers_per_tour)
    
    # Calculate number of clusters - each cluster should have no more than max_cluster_size containers
    num_clusters = math.ceil(total_containers / max_cluster_size)
    
    # Ensure we don't create empty clusters
    num_clusters = min(num_clusters, total_containers)
    
    logger.info(f"Creating {num_clusters} clusters for {needed_tours} tours")
    
    # Apply clustering based on chosen method
    if clustering_method == 'kmeans':
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
    elif clustering_method == 'dbscan':
        # Apply DBSCAN clustering - need to tune eps and min_samples
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(embedding_matrix)
        
        # Handle any unclustered containers (label -1)
        if -1 in cluster_labels:
            # Find nearest cluster for each unclustered container
            unclustered = np.where(cluster_labels == -1)[0]
            logger.warning(f"DBSCAN produced {len(unclustered)} unclustered containers")
            
            # Assign each unclustered container to nearest cluster
            for idx in unclustered:
                if np.any(cluster_labels >= 0):  # Make sure there are clustered points
                    clustered_indices = np.where(cluster_labels >= 0)[0]
                    distances = pairwise_distances(
                        embedding_matrix[idx].reshape(1, -1),
                        embedding_matrix[clustered_indices]
                    )[0]
                    closest_idx = clustered_indices[np.argmin(distances)]
                    cluster_labels[idx] = cluster_labels[closest_idx]
                else:
                    # If all are unclustered, create a new cluster
                    cluster_labels[idx] = max(np.max(cluster_labels) + 1, 0)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    
    # Create initial clusters
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[str(label)].append(container_ids[i])
    
    # Balance clusters to avoid exceeding max_cluster_size
    balanced_clusters = balance_clusters(clusters, max_cluster_size)
    
    # Ensure all critical containers are included
    final_clusters = ensure_critical_containers(balanced_clusters, critical_containers, container_embeddings, max_cluster_size)
    
    return final_clusters

def balance_clusters(clusters: Dict[str, List[str]], max_cluster_size: int) -> Dict[str, List[str]]:
    """
    Balance cluster sizes to ensure none exceed max_cluster_size.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    max_cluster_size : int
        Maximum allowed cluster size
        
    Returns
    -------
    Dict[str, List[str]]
        Balanced clusters
    """
    # Check if any clusters exceed the maximum size
    oversized_clusters = {
        cluster_id: containers
        for cluster_id, containers in clusters.items()
        if len(containers) > max_cluster_size
    }
    
    if not oversized_clusters:
        return clusters  # No balancing needed
    
    logger.info(f"Balancing {len(oversized_clusters)} oversized clusters")
    
    # Create a copy of the clusters to modify
    balanced = {k: v.copy() for k, v in clusters.items()}
    
    # Process each oversized cluster
    for cluster_id, containers in oversized_clusters.items():
        # Calculate how many new clusters to create
        num_containers = len(containers)
        num_subclusters = math.ceil(num_containers / max_cluster_size)
        
        if num_subclusters <= 1:
            continue  # This shouldn't happen, but just in case
            
        # Create subclusters with kmeans
        container_groups = np.array_split(containers, num_subclusters)
        
        # Replace original cluster with first subcluster
        balanced[cluster_id] = container_groups[0].tolist()
        
        # Add new subclusters
        for i in range(1, num_subclusters):
            new_cluster_id = f"{cluster_id}_{i}"
            balanced[new_cluster_id] = container_groups[i].tolist()
    
    # Check if any new clusters are still oversized
    still_oversized = any(len(containers) > max_cluster_size for containers in balanced.values())
    if still_oversized:
        # Recursively balance until no oversized clusters remain
        return balance_clusters(balanced, max_cluster_size)
    
    return balanced

def ensure_critical_containers(clusters: Dict[str, List[str]], 
                              critical_containers: Set[str],
                              container_embeddings: Dict[str, np.ndarray],
                              max_cluster_size: int) -> Dict[str, List[str]]:
    """
    Ensure all critical containers are assigned to clusters.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    critical_containers : Set[str]
        Set of critical container IDs
    container_embeddings : Dict[str, np.ndarray]
        Dictionary mapping container IDs to embedding vectors
    max_cluster_size : int
        Maximum allowed cluster size
        
    Returns
    -------
    Dict[str, List[str]]
        Clusters with all critical containers assigned
    """
    # Check if any critical containers are missing from clusters
    all_assigned_containers = set()
    for containers in clusters.values():
        all_assigned_containers.update(containers)
    
    missing_critical = critical_containers - all_assigned_containers
    
    if not missing_critical:
        return clusters  # All critical containers are already assigned
    
    logger.info(f"Ensuring {len(missing_critical)} missing critical containers are assigned")
    
    # Create a copy of the clusters to modify
    updated_clusters = {k: v.copy() for k, v in clusters.items()}
    
    # For each missing critical container, assign to most similar cluster
    for container_id in missing_critical:
        # Skip if container has no embedding
        if container_id not in container_embeddings:
            logger.warning(f"Critical container {container_id} has no embedding, cannot assign")
            continue
            
        critical_embedding = container_embeddings[container_id]
        
        # Find best cluster for this container
        best_cluster_id = None
        best_similarity = -float('inf')
        
        for cluster_id, cluster_containers in updated_clusters.items():
            # Skip full clusters
            if len(cluster_containers) >= max_cluster_size:
                continue
                
            # Calculate average similarity to containers in this cluster
            if not cluster_containers:
                continue
                
            similarities = []
            for other_id in cluster_containers:
                if other_id in container_embeddings:
                    other_embedding = container_embeddings[other_id]
                    # Use cosine similarity
                    similarity = np.dot(critical_embedding, other_embedding) / (
                        np.linalg.norm(critical_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster_id = cluster_id
        
        # If found a suitable cluster, add the container
        if best_cluster_id is not None:
            updated_clusters[best_cluster_id].append(container_id)
        else:
            # Create a new cluster for this critical container
            new_cluster_id = f"critical_{len(updated_clusters)}"
            updated_clusters[new_cluster_id] = [container_id]
    
    return updated_clusters

# Drop-in compatible replacement for the original cluster_containers() function
def cluster_containers(container_data: pd.DataFrame, slotbook_data: pd.DataFrame, 
                      batch_size: Optional[int] = None, 
                      containers_per_tour: int = 20,
                      max_cluster_size: int = 200,
                      use_distinct_aisles: bool = True, 
                      min_clusters: int = 2, 
                      max_clusters: int = 10,
                      generate_visuals: bool = False, 
                      output_path: str = './cluster_analysis') -> Dict[str, List[str]]:
    """
    Drop-in replacement for the original cluster_containers function that uses GNN instead of hierarchical clustering.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    batch_size : Optional[int]
        Optional number of tours to form in each batch
    containers_per_tour : int
        Maximum containers per tour
    max_cluster_size : int
        Maximum size for any cluster
    use_distinct_aisles : bool
        Whether to use distinct aisles (True) or aisle span (False) as secondary feature - ignored in GNN approach
    min_clusters : int
        Minimum number of clusters to consider - ignored in GNN approach
    max_clusters : int
        Maximum number of clusters to consider - ignored in GNN approach
    generate_visuals : bool
        Whether to generate visualizations - ignored in GNN approach
    output_path : str
        Path to save visualizations - used for model saving
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    logger.info("Using GNN-based container clustering instead of hierarchical clustering")
    
    # Call our GNN-based clustering with compatible parameters
    return gnn_cluster_containers(
        container_data=container_data,
        slotbook_data=slotbook_data,
        batch_size=batch_size,
        containers_per_tour=containers_per_tour,
        max_cluster_size=max_cluster_size,
        use_pretrained=True,  # Try to use pretrained model if available
        model_path=os.path.join(output_path, 'container_gnn.pt')
    )

def evaluate_clusters(clusters: Dict[str, List[str]], container_data: pd.DataFrame, 
                     slotbook_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate the quality of generated clusters.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of cluster quality metrics
    """
    logger.info("Evaluating cluster quality...")
    
    # Initialize metrics
    metrics = {
        'cluster_count': len(clusters),
        'total_containers': sum(len(containers) for containers in clusters.values()),
        'cluster_sizes': {},
        'aisle_metrics': {},
        'critical_container_metrics': {},
        'overall': {}
    }
    
    # Create SKU to aisle mapping
    sku_aisle_mapping = create_sku_aisle_mapping(slotbook_data)
    
    # Extract critical containers
    critical_containers = get_critical_containers(container_data)
    critical_count = len(critical_containers)
    
    # Calculate cluster size distribution
    cluster_sizes = [len(containers) for containers in clusters.values()]
    metrics['cluster_sizes'] = {
        'min': min(cluster_sizes) if cluster_sizes else 0,
        'max': max(cluster_sizes) if cluster_sizes else 0,
        'avg': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
        'distribution': Counter(cluster_sizes)
    }
    
    # Calculate aisle metrics for each cluster
    cluster_aisle_metrics = {}
    total_aisle_span = 0
    total_distinct_aisles = 0
    
    for cluster_id, container_ids in clusters.items():
        # Calculate min/max aisles and distinct aisles
        all_aisles = set()
        for container_id in container_ids:
            container_aisles = get_container_aisles(container_id, container_data, sku_aisle_mapping)
            all_aisles.update(container_aisles)
        
        if all_aisles:
            min_aisle = min(all_aisles)
            max_aisle = max(all_aisles)
            aisle_span = max_aisle - min_aisle
            distinct_aisles = len(all_aisles)
        else:
            min_aisle = 0
            max_aisle = 0
            aisle_span = 0
            distinct_aisles = 0
            
        cluster_aisle_metrics[cluster_id] = {
            'min_aisle': min_aisle,
            'max_aisle': max_aisle,
            'aisle_span': aisle_span,
            'distinct_aisles': distinct_aisles
        }
        
        total_aisle_span += aisle_span
        total_distinct_aisles += distinct_aisles
    
    # Calculate average metrics
    metrics['aisle_metrics'] = {
        'avg_aisle_span': total_aisle_span / len(clusters) if clusters else 0,
        'avg_distinct_aisles': total_distinct_aisles / len(clusters) if clusters else 0,
        'cluster_details': cluster_aisle_metrics
    }
    
    # Calculate critical container metrics
    critical_coverage = 0
    critical_distribution = []
    
    for cluster_id, container_ids in clusters.items():
        critical_in_cluster = sum(1 for c_id in container_ids if c_id in critical_containers)
        critical_coverage += critical_in_cluster
        if critical_in_cluster > 0:
            critical_distribution.append(critical_in_cluster)
    
    metrics['critical_container_metrics'] = {
        'total_critical': critical_count,
        'critical_coverage': critical_coverage,
        'critical_coverage_percent': (critical_coverage / critical_count * 100) if critical_count > 0 else 0,
        'critical_distribution': Counter(critical_distribution),
        'clusters_with_critical': len(critical_distribution)
    }
    
    # Calculate overall metrics combining multiple factors
    if critical_count > 0:
        critical_score = critical_coverage / critical_count
    else:
        critical_score = 1.0  # No critical containers to cover
    
    aisle_efficiency = 1.0 / (metrics['aisle_metrics']['avg_aisle_span'] / 10 + 1)
    size_balance = 1.0 - (metrics['cluster_sizes']['max'] - metrics['cluster_sizes']['avg']) / metrics['cluster_sizes']['max'] if metrics['cluster_sizes']['max'] > 0 else 0
    
    # Combined score (weighted average)
    combined_score = (0.5 * critical_score + 0.3 * aisle_efficiency + 0.2 * size_balance)
    
    metrics['overall'] = {
        'critical_score': critical_score,
        'aisle_efficiency': aisle_efficiency,
        'size_balance': size_balance,
        'combined_score': combined_score
    }
    
    # Log summary
    logger.info(f"Evaluation results:")
    logger.info(f"- Cluster count: {metrics['cluster_count']}")
    logger.info(f"- Total containers: {metrics['total_containers']}")
    logger.info(f"- Critical coverage: {metrics['critical_container_metrics']['critical_coverage_percent']:.1f}%")
    logger.info(f"- Avg aisle span: {metrics['aisle_metrics']['avg_aisle_span']:.1f}")
    logger.info(f"- Avg distinct aisles: {metrics['aisle_metrics']['avg_distinct_aisles']:.1f}")
    logger.info(f"- Combined score: {metrics['overall']['combined_score']:.4f}")
    
    return metrics

def gnn_cluster_containers(container_data: pd.DataFrame, slotbook_data: pd.DataFrame, 
                          batch_size: Optional[int] = None, 
                          containers_per_tour: int = 20,
                          max_cluster_size: int = 200,
                          use_pretrained: bool = False,
                          model_path: str = './container_gnn.pt',
                          device: str = None) -> Dict[str, List[str]]:
    """
    Main entry point for GNN-based container clustering.
    
    This function replaces the hierarchical clustering approach with GNN-based clustering
    but maintains the same interface for compatibility with the existing codebase.
    
    Parameters
    ----------
    container_data : pd.DataFrame
        Container data with order details
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    batch_size : Optional[int]
        Optional number of tours to form in each batch (if None, determined automatically)
    containers_per_tour : int
        Number of containers per tour
    max_cluster_size : int
        Maximum allowed cluster size
    use_pretrained : bool
        Whether to use a pretrained model if available
    model_path : str
        Path to save/load the GNN model
    device : str
        Device to use for training and inference
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping cluster IDs to lists of container IDs
    """
    start_time = time.time()
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Starting GNN-based container clustering on {device}...")
    logger.info(f"Processing {len(container_data['container_id'].unique())} containers")
    
    # Build container graph
    container_graph = build_container_graph(container_data, slotbook_data)
    
    # Create SKU to aisle mapping (needed for pair generation)
    sku_aisle_mapping = create_sku_aisle_mapping(slotbook_data)
    
    # Load or train GNN model
    if use_pretrained and os.path.exists(model_path):
        logger.info(f"Loading pretrained GNN model from {model_path}")
        gnn_model = torch.load(model_path, map_location=device)
    else:
        # Train new model
        gnn_model = train_gnn(container_graph, container_data, sku_aisle_mapping, device=device)
        
        # Save model if path is provided
        if model_path:
            logger.info(f"Saving trained GNN model to {model_path}")
            torch.save(gnn_model, model_path)
    
    # Generate embeddings
    container_embeddings = generate_embeddings(gnn_model, container_graph, device=device)
    
    # Cluster with constraints
    clusters = cluster_with_constraints(
        container_embeddings,
        container_data,
        max_cluster_size=max_cluster_size,
        containers_per_tour=containers_per_tour
    )
    
    # Evaluate clusters
    metrics = evaluate_clusters(clusters, container_data, slotbook_data)
    
    # Log results
    total_time = time.time() - start_time
    logger.info(f"GNN-based clustering completed in {total_time:.2f} seconds")
    logger.info(f"Created {len(clusters)} clusters")
    
    # Validate cluster sizes
    cluster_sizes = [len(containers) for containers in clusters.values()]
    max_size = max(cluster_sizes) if cluster_sizes else 0
    min_size = min(cluster_sizes) if cluster_sizes else 0
    avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
    
    logger.info(f"Cluster statistics: min={min_size}, max={max_size}, avg={avg_size:.1f}")
    
    return clusters