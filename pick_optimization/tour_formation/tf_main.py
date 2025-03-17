"""
Main entry point for tour formation.

This module provides the main entry point for the tour formation process - 
includes data load, global and clustered solving.
"""

from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

from .tf_solver import TourFormationSolver, TourFormationResult
from .clustering.clusterer import ContainerClusterer

def solve_cluster(
    cluster_id: str,
    container_ids: List[str],
    containers_df: pd.DataFrame,
    slotbook_data: pd.DataFrame,
    planning_timestamp: datetime,
    config: Dict[str, Any],
    logger: logging.Logger,
    tour_id_offset: int = 0,
    num_tours: int = 1
) -> Optional[TourFormationResult]:
    """
    Solve tour formation for a single cluster.
    
    Parameters
    ----------
    cluster_id : str
        Identifier for the cluster
    container_ids : List[str]
        List of container IDs in this cluster
    containers_df : pd.DataFrame
        Full container data
    slotbook_data : pd.DataFrame
        Slotbook data with inventory details
    planning_timestamp : datetime
        Current planning timestamp
    config : Dict[str, Any]
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    tour_id_offset : int
        Offset for tour IDs
    num_tours : int
        Number of tours to form for this cluster
        
    Returns
    -------
    Optional[TourFormationResult]
        Solution for this cluster if found
    """
    try:
        # Create solver instance
        solver = TourFormationSolver(
            container_data=containers_df,
            slotbook_data=slotbook_data,
            planning_timestamp=planning_timestamp,
            config=config,
            num_tours=num_tours  
        )
        
        # Prepare data for this specific cluster
        solver.prepare_data(container_ids=container_ids)
        
        # Set the tour ID offset
        solver.tour_id_offset = tour_id_offset
        
        # Solve the problem
        solution = solver.solve()
        
        if solution:
            # Adjust tour IDs with offset if needed
            if tour_id_offset > 0:
                # Update container assignments
                for container_data in solution['container_assignments'].values():
                    if 'tour' in container_data:
                        container_data['tour'] += tour_id_offset
                
                # Update aisle ranges
                updated_aisle_ranges = {}
                for tour_id, aisle_range in solution['aisle_ranges'].items():
                    new_tour_id = tour_id + tour_id_offset
                    updated_aisle_ranges[new_tour_id] = aisle_range
                solution['aisle_ranges'] = updated_aisle_ranges
            
            return TourFormationResult(
                container_assignments=solution['container_assignments'],
                pick_assignments=solution['pick_assignments'],
                aisle_ranges=solution['aisle_ranges'],
                metrics=solution['metrics'],
                solve_time=solution.get('solve_time', 0.0),
                cluster_id=cluster_id
            )
        else:
            logger.warning(f"No solution found for cluster {cluster_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error solving cluster {cluster_id}: {str(e)}")
        return None


def run_tour_formation(
    containers_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime = None
) -> List[TourFormationResult]:
    """
    Run the tour formation process.
    
    Parameters
    ----------
    containers_df : pd.DataFrame
        DataFrame containing container information
    skus_df : pd.DataFrame
        DataFrame containing SKU information
    config : Dict[str, Any]
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    planning_timestamp : datetime, optional
        Current planning timestamp
        
    Returns
    -------
    List[TourFormationResult]
        List of results from each iteration
    """
    try:
        
        # Check if clustering is enabled
        use_clustering = config['tour_formation']['clustering_enabled']
        
        if use_clustering:
            # Initialize clusterer
            clusterer = ContainerClusterer(config, logger)
            
            # Get clusters
            clusters, cluster_tours = clusterer.cluster_containers(containers_df, skus_df)
            
            # Prepare for parallel execution if enabled
            use_parallel = config.get('tour_formation', {}).get('parallel_execution', False)
            
            if use_parallel:
                # Solve clusters in parallel
                # TODO: Implement parallel execution
                logger.info("Parallel execution not implemented yet. Using sequential execution.")
                raise NotImplementedError("Parallel execution not implemented yet.")
            else:
                # Solve clusters sequentially
                results = []
                tour_id_offset = 0
                
                for cluster_id, container_ids in clusters.items():
                    # Skip empty clusters
                    if not container_ids:
                        continue
                    
                    logger.info(f"Starting tour formation for cluster {cluster_id} with {len(container_ids)} containers")  
                      
                    # Skip clusters that are too small
                    if len(container_ids) < config['tour_formation']['min_containers_per_tour']:
                        logger.warning(f"Cluster {cluster_id} with {len(container_ids)} containers is too small to form a tour. Skipping.")
                        continue
                    
                    result = solve_cluster(
                        cluster_id,
                        container_ids,
                        containers_df,
                        skus_df,
                        planning_timestamp,
                        config,
                        logger,
                        tour_id_offset,
                        num_tours=cluster_tours.get(cluster_id)  # Pass number of tours needed
                    )
                    
                    if result:
                        results.append(result)
                        tour_id_offset += cluster_tours.get(cluster_id)
                
        else:
            # Solve globally without clustering
            solver = TourFormationSolver(config, logger)
            result = solver.solve(
                containers_df=containers_df,
                skus_df=skus_df
            )
            results = [result] if result else []
        
        # Process and log results
        process_and_log_results(results, logger)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in tour formation process: {str(e)}")
        raise 
    
def process_and_log_results(results: List[TourFormationResult], logger: logging.Logger) -> None:
    """
    Process tour formation results and log detailed summary.
    
    Parameters
    ----------
    results : List[TourFormationResult]
        List of results from tour formation
    logger : logging.Logger
        Logger instance
    """
    if not results:
        logger.warning("No solutions found in tour formation")
        return
        
    # Calculate overall metrics
    total_containers = sum(len(r.container_assignments) for r in results)
    total_tours = sum(len(r.aisle_ranges) for r in results)
    total_slack = sum(r.metrics['total_lateness'] for r in results)
    total_aisle_span = sum(r.metrics['total_distance'] for r in results)
    total_time = sum(r.solve_time for r in results)
    
    # Calculate tour-specific metrics
    tour_metrics = []
    total_units = 0
    total_unique_aisles = 0
    
    for result in results:
        cluster = result.cluster_id
        for tour_id, aisle_range in result.aisle_ranges.items():
            # Get containers for this specific tour
            tour_containers = [
                c for c, info in result.container_assignments.items()
                if info['tour'] == tour_id
            ]
            
            # Get picks for these containers
            tour_picks = []
            for container in tour_containers:
                if container in result.pick_assignments:
                    tour_picks.extend(result.pick_assignments[container])
            
            # Calculate metrics for this tour
            tour_units = sum(p['quantity'] for p in tour_picks)
            total_units += tour_units
            
            unique_aisles = len(set(p['aisle'] for p in tour_picks))
            total_unique_aisles += unique_aisles
            
            aisle_span = aisle_range['max_aisle'] - aisle_range['min_aisle']
            
            tour_metrics.append({
                'tour_id': tour_id,
                'cluster': cluster,
                'num_containers': len(tour_containers),
                'num_picks': len(tour_picks),
                'units': tour_units,
                'aisle_span': aisle_span,
                'distinct_aisles': unique_aisles,
                'slack': result.metrics.get('total_lateness', 0)
            })
    
    # Calculate averages
    if tour_metrics:
        avg_containers = np.mean([m['num_containers'] for m in tour_metrics])
        avg_picks = np.mean([m['num_picks'] for m in tour_metrics])
        avg_units = np.mean([m['units'] for m in tour_metrics])
        avg_aisle_span = np.mean([m['aisle_span'] for m in tour_metrics])
        avg_distinct_aisles = np.mean([m['distinct_aisles'] for m in tour_metrics])
    else:
        avg_containers = avg_picks = avg_units = avg_aisle_span = avg_distinct_aisles = 0
    
    # Log overall summary
    logger.info("\nTour Formation Summary")
    
    overall_table = [
        ["Metric", "Value"],
        ["Total Containers", f"{total_containers}"],
        ["Total Tours", f"{total_tours}"],
        ["Total Units", f"{total_units}"],
        ["Total Aisles In", f"{total_unique_aisles}"],
        ["Total Aisle Across", f"{total_aisle_span}"],
        ["Total Slack", f"{total_slack:.2f} hours"],
        ["Total Solve Time", f"{total_time:.2f} seconds"]
    ]
    logger.info("\n" + tabulate(overall_table, headers="firstrow", tablefmt="grid"))
    
    # Log averages
    logger.info("\nPer Tour Averages:")
    averages_table = [
        ["Metric", "Value"],
        ["Containers", f"{avg_containers:.2f}"],
        ["Picks", f"{avg_picks:.2f}"],
        ["Units", f"{avg_units:.2f}"],
        ["Aisles In", f"{avg_distinct_aisles:.2f}"],
        ["Aisle Across", f"{avg_aisle_span:.2f}"]
    ]
    logger.info("\n" + tabulate(averages_table, headers="firstrow", tablefmt="grid"))
    
    # Log tour-specific details
    logger.info("\nTour Details:")
    tour_details = []
    headers = ["Tour ID", "Cluster", "Containers", "Picks", "Units", "Aisles In", "Aisle Across", "Slack"]
    
    for metrics in sorted(tour_metrics, key=lambda x: (x['cluster'], x['tour_id'])):
        tour_details.append([
            metrics['tour_id'],
            metrics['cluster'],
            metrics['num_containers'],
            metrics['num_picks'],
            metrics['units'],
            metrics['distinct_aisles'],
            metrics['aisle_span'],
            f"{metrics['slack']:.2f}"
        ])
    
    logger.info("\n" + tabulate(tour_details, headers=headers, tablefmt="grid", showindex=False))

