"""
Orchestrator for the tour formation process.

Coordinates the steps of clustering, solving, and component initialization
based on the specified execution mode.
"""

# Standard library imports
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Third-party imports
import pandas as pd

# Local imports
from .slack_calculator import SlackCalculator
from .tf_solver_service import TourFormationSolverService
from .clustering.clusterer import ContainerClusterer
from .tf_data_validator import DataValidator
from .data_exchange import load_cached_containers_with_slack, write_cached_containers_with_slack

# Get module-specific logger
logger = logging.getLogger(__name__)

def calculate_average_upc(containers_df: pd.DataFrame, logger: logging.Logger) -> float:
    """
    Calculate the average units per container (UPC) from containers data.
    
    Parameters
    ----------
    containers_df : pd.DataFrame
        DataFrame containing container data with pick_quantity column
    logger : logging.Logger
        Logger instance for logging messages
        
    Returns
    -------
    float
        Average UPC value
    """
    # Calculate total pick quantity
    total_pick_quantity = containers_df['pick_quantity'].sum()
    
    # Calculate average UPC
    avg_upc = total_pick_quantity / len(containers_df['container_id'].unique())
    
    logger.info(f"Calculated average UPC: {avg_upc:.2f}")
    return avg_upc

def calculate_container_target(
    active_pickers: int,
    avg_pick_uph: float,
    avg_container_upc: float,
    variability_factor: float,
    logger: logging.Logger
) -> int:
    """
    Calculate the target number of containers to process per interval.
    
    Parameters:
    -----------
    active_pickers : int
        Current active picker headcount (R)
    avg_pick_uph : float
        Historical average pick units per hour (U)
    avg_container_upc : float
        Average backlog container units per container (C)
    variability_factor : float
        Factor to account for demand fluctuations and uncertainties (gamma)
    logger : logging.Logger
        Logger instance for logging messages
    
    Returns:
    --------
    int
        Target number of containers per interval, rounded to nearest integer
    
    Notes:
    ------
    The formula used is: Tfinal = gamma * R * (U / C)
    """
    # Step 1: Base target calculation
    base_target = active_pickers * (avg_pick_uph / avg_container_upc)
    
    # Step 2: Adjust for variability
    final_target = variability_factor * base_target
    
    final_target = round(final_target)
    
    logger.info(f"Calculated container target: {final_target}")
    
    # Return as integer (rounded)
    return final_target

def _create_components(
    containers_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime,
    output_dir: str,
    working_dir: str,
    mode: str,
    labor_headcount: int
) -> Dict[str, Any]:
    """
    Create and initialize components needed for tour formation.
    Includes data validation and loads/saves container slack calculation based on mode.
    """
    containers_df_with_slack = None
    calculated_slack = False
    slack_calculator = None

    # Calculate container target
    avg_upc = calculate_average_upc(containers_df, logger)
    container_target = calculate_container_target(
        active_pickers=labor_headcount,
        avg_pick_uph=config['global']['avg_pick_uph'],
        avg_container_upc=avg_upc,
        variability_factor=config['global']['container_target_variability_factor'],
        logger=logger
    )
    
    # 1. Loading from cache if in solve_cluster mode
    if mode == 'solve_cluster':
        containers_df_with_slack = load_cached_containers_with_slack(working_dir, logger)
    else:
        containers_df_with_slack = None

    # 2. Calculate slack if not loaded from cache
    if containers_df_with_slack is None:
        slack_calculator = SlackCalculator(config=config, logger=logger)
        if 'slack_category' not in containers_df.columns:
            logger.info("Calculating container slack (orchestrator)")
            containers_df_with_slack = slack_calculator.calculate_container_slack(
                container_data=containers_df,
                current_time=planning_timestamp,
                slotbook_data=skus_df,
                labor_headcount=labor_headcount,
                container_target=container_target
            )
            calculated_slack = True
        else:
            logger.info("Slack category already present in input container data.")
            containers_df_with_slack = containers_df

    # 3. Save to cache if slack was calculated and not in solve_cluster mode
    if calculated_slack and mode != 'solve_cluster':
        write_cached_containers_with_slack(
            working_dir=working_dir,
            containers_df=containers_df_with_slack,
            logger=logger
        )

    # --- Data Validation Step ---
    # Validate data after slack calculation
    logger.info("Performing data validation")
    validator = DataValidator(logger=logger)
    try:
        # Validate and update the dataframes
        containers_df_with_slack, skus_df = validator.validate(
            containers_df_with_slack, 
            skus_df
        )
        logger.info("Data validation successful.")
    except Exception as e:
        logger.error(f"Data validation failed: {e}", exc_info=True)
        raise

    # --- Initialize other components ---
    # Pass output_dir to ContainerClusterer initialization
    clusterer = ContainerClusterer(config=config, logger=logger, output_dir=output_dir, container_target=container_target)

    # Initialize the Solver Service
    solver_service = TourFormationSolverService(config=config, logger=logger, output_dir=output_dir)

    return {
        'slack_calculator': slack_calculator,      
        'clusterer': clusterer,
        'solver_service': solver_service,
        'containers_df': containers_df_with_slack,  # Use the final df (loaded or calculated)
        'skus_df': skus_df,
        'output_dir': output_dir,
        'working_dir': working_dir
    }


def _perform_clustering(
    clusterer: ContainerClusterer,
    containers_df: pd.DataFrame,
    skus_df: pd.DataFrame, 
    logger: logging.Logger
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Perform container clustering."""
    try:
        logger.info("Performing container clustering")
      
        clusters, cluster_tours = clusterer.cluster_containers(
            container_data=containers_df,
            slotbook_data=skus_df
        )
        logger.info(f"Clustering complete. Found {len(clusters)} clusters.")
        
        # --- Create Cluster Metadata --- 
        # The clusterer itself likely holds metadata or can generate it.
        # We need metadata (like tour_id_offset) for the solver service.
        # Assuming the clusterer class has a method or property for this.

        # TODO: Refine how cluster_metadata (esp. tour_id_offset) is obtained
        cluster_metadata = clusterer.cluster_metadata if hasattr(clusterer, 'cluster_metadata') else {}

        return clusters, cluster_metadata
    
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise


# --- Mode-Specific Workflow Functions ---

def run_local_workflow(
    components: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime,
    labor_headcount: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run the full local workflow: clustering + solving all clusters."""
    logger.info("Running local workflow (clustering + solving)")
    clusterer = components['clusterer']
    solver_service: TourFormationSolverService = components['solver_service']
    containers_df = components['containers_df'] # Use potentially updated df
    skus_df = components['skus_df']

    clusters, cluster_metadata = _perform_clustering(
        clusterer=clusterer,
        containers_df=containers_df,
        skus_df=skus_df,
        logger=logger
    )

    results = []
    for cluster_id, container_ids in clusters.items():
        # Use the solver service to solve each cluster
        result = solver_service.solve_one_cluster(
            container_ids=container_ids,
            containers_df=containers_df,
            skus_df=skus_df,
            planning_timestamp=planning_timestamp,
            cluster_id=cluster_id,
            cluster_metadata=cluster_metadata.get(cluster_id, {})
        )
        # Result structure from service already includes status, metadata etc.
        results.append(result) # Append result regardless of status

    # Result aggregation/logging can happen here or upstream
    successful_solves = [r for r in results if r.get('status') == 'Optimal' or r.get('status') == 'Feasible']
    skipped_solves = [r for r in results if r.get('status') == 'Skipped']
    failed_solves = [r for r in results if r.get('status') not in ['Optimal', 'Feasible', 'Skipped']]
    
    logger.info(
        f"Local workflow finished. Total clusters: {len(results)}. "
        f"Successful: {len(successful_solves)}, Skipped: {len(skipped_solves)}, Failed: {len(failed_solves)}."
    )
    return results


def run_clustering_step(
    components: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Run only the clustering step."""
    
    logger.info("Running clustering step")
    clusterer = components['clusterer']
    containers_df = components['containers_df']
    skus_df = components['skus_df']

    clusters, cluster_metadata = _perform_clustering(
        clusterer=clusterer,
        containers_df=containers_df,
        skus_df=skus_df,
        logger=logger
    )
    
    return clusters, cluster_metadata


def solve_single_cluster(
    components: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime,
    cluster_id: str,
    container_ids: List[str],
    cluster_metadata: Dict[str, Any],
    labor_headcount: Optional[int] = None
) -> Dict[str, Any]:
    """
    Solves a single, specified cluster using the Solver Service.
    """
    
    logger.info(f"Orchestrating solve for single cluster: {cluster_id}")
    solver_service: TourFormationSolverService = components['solver_service']
    containers_df = components['containers_df']
    skus_df = components['skus_df']

    # Call the solver service
    result = solver_service.solve_one_cluster(
        container_ids=container_ids,
        containers_df=containers_df,
        skus_df=skus_df,
        planning_timestamp=planning_timestamp,
        cluster_id=cluster_id,
        cluster_metadata=cluster_metadata
    )
    return result


# --- Main Orchestration Function ---

def orchestrate_tour_formation(
    containers_df: pd.DataFrame,
    skus_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime,
    mode: str,
    output_dir: str,
    working_dir: str,
    cluster_to_solve_id: Optional[str] = None,
    cluster_to_solve_containers: Optional[List[str]] = None,
    cluster_to_solve_metadata: Optional[Dict[str, Any]] = None,
    labor_headcount: Optional[int] = None 
) -> Any:
    """
    Orchestrates the tour formation process based on the mode.

    Returns:
        - List[Dict]: Results for 'run_complete' mode.
        - Tuple[Dict, Dict]: Clusters and metadata for 'generate_clusters' mode.
        - Dict: Result for a single cluster in 'solve_cluster' mode.
        - None: If mode is invalid or an error occurs upstream.
    """
    valid_modes = ['run_complete', 'generate_clusters', 'solve_cluster']
    if mode not in valid_modes:
        logger.error(f"Invalid mode specified for orchestrator: {mode}")
        raise ValueError(f"Invalid mode: {mode}. Expected one of {valid_modes}")

    # Initialize components
    components = _create_components(
        containers_df=containers_df,
        skus_df=skus_df,
        config=config,
        logger=logger,
        planning_timestamp=planning_timestamp,
        output_dir=output_dir,
        working_dir=working_dir,
        mode=mode,  
        labor_headcount=labor_headcount
    )

    # --- Mode-Specific Execution ---
    if mode == 'run_complete':
        return run_local_workflow(
            components=components,
            config=config,
            logger=logger,
            planning_timestamp=planning_timestamp,
            labor_headcount=labor_headcount
        )
    elif mode == 'generate_clusters':
        return run_clustering_step(
            components=components,
            logger=logger
        )
    elif mode == 'solve_cluster':
        if not all([cluster_to_solve_id, cluster_to_solve_containers is not None, 
                    cluster_to_solve_metadata is not None]):
             error_msg = "Missing required data for 'solve_cluster' mode in orchestrator."
             logger.error(error_msg)
             raise ValueError(error_msg)

        return solve_single_cluster(
            components=components,
            config=config,
            logger=logger,
            planning_timestamp=planning_timestamp,
            cluster_id=cluster_to_solve_id,
            container_ids=cluster_to_solve_containers,
            cluster_metadata=cluster_to_solve_metadata,
            labor_headcount=labor_headcount
        )
    else:
        logger.error(f"Reached unexpected point in orchestrator for mode: {mode}")
        return None 
