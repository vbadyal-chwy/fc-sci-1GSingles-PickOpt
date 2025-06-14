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
from .data_exchange import load_cached_containers_with_slack, write_cached_containers_with_slack, save_validated_data, load_validated_data

# Get module-specific logger with workflow logging
from pick_optimization.utils.logging_config import get_logger
logger = get_logger(__name__, 'tour_formation')

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
    wh_id: str,
    planning_timestamp: datetime,
    active_pickers: int,
    avg_pick_uph: float,
    avg_container_upc: float,
    variability_factor: float,
    containers_df: pd.DataFrame,
    tf_min_containers_in_backlog: int,
    logger: logging.Logger
) -> Tuple[int, pd.DataFrame]:
    """
    Calculate the target number of containers to process per interval and evaluate TF trigger logic.
    
    Parameters:
    -----------
    wh_id : str
        Warehouse ID
    planning_timestamp : datetime
        Planning timestamp for this calculation
    active_pickers : int
        Current active picker headcount (R)
    avg_pick_uph : float
        Historical average pick units per hour (U)
    avg_container_upc : float
        Average backlog container units per container (C)
    variability_factor : float
        Factor to account for demand fluctuations and uncertainties (gamma)
    containers_df : pd.DataFrame
        DataFrame containing current container backlog data
    tf_min_containers_in_backlog : int
        Minimum number of containers required to trigger TF
    logger : logging.Logger
        Logger instance for logging messages
    
    Returns:
    --------
    Tuple[int, pd.DataFrame]
        - Target number of containers per interval, rounded to nearest integer
        - DataFrame containing input parameters, calculated target, and TF trigger evaluation
    
    Notes:
    ------
    The formula used is: Tfinal = gamma * R * (U / C)
    TF trigger logic: trigger_tf_flag = (backlog_container_count >= tf_min_containers_in_backlog)
    """
    # Step 1: Calculate backlog container count
    backlog_container_count = containers_df['container_id'].nunique()
    logger.info(f"Backlog container count: {backlog_container_count}")
    
    # Step 2: Evaluate TF trigger logic
    trigger_tf_flag = backlog_container_count >= tf_min_containers_in_backlog
    logger.info(f"TF trigger evaluation: {backlog_container_count} >= {tf_min_containers_in_backlog} = {trigger_tf_flag}")
    
    if not trigger_tf_flag:
        logger.info(f"TF will be skipped: backlog container count ({backlog_container_count}) "
                   f"is below minimum threshold ({tf_min_containers_in_backlog})")
    
    # Step 3: Base target calculation
    base_target = active_pickers * (avg_pick_uph / avg_container_upc)
    
    # Step 4: Adjust for variability
    final_target = variability_factor * base_target
    
    final_target = round(final_target)
    
    logger.info(f"Calculated container target: {final_target}")
    
    # Create metrics DataFrame with actual values
    container_target_df = pd.DataFrame({
        'wh_id': [wh_id],
        'planning_datetime': [planning_timestamp],  # Use datetime for database compatibility
        'active_headcount_multis': [active_pickers],
        'historical_uph_multis': [avg_pick_uph],
        'avg_upc_multis': [avg_container_upc],
        'container_target_variability_factor': [variability_factor],
        'target_containers_per_interval': [final_target],
        'backlog_container_count': [backlog_container_count],  # Actual backlog count
        'trigger_tf_flag': [trigger_tf_flag]  # Actual trigger evaluation
    })
    
    # Return as tuple of integer (rounded) and metrics DataFrame
    return final_target, container_target_df


def should_trigger_tf(container_target_df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Check if TF should be triggered based on the container target evaluation.
    
    Parameters
    ----------
    container_target_df : pd.DataFrame
        DataFrame containing TF trigger evaluation results
    logger : logging.Logger
        Logger instance for logging messages
        
    Returns
    -------
    bool
        True if TF should be triggered, False otherwise
    """

    trigger_flag = container_target_df['trigger_tf_flag'].iloc[0]
    backlog_count = container_target_df.get('backlog_container_count', pd.Series([0])).iloc[0]
    
    logger.info(f"TF trigger check: trigger_tf_flag={trigger_flag}, backlog_container_count={backlog_count}")
    return bool(trigger_flag)

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

    # Calculate container target and evaluate TF trigger logic (skip for solve_cluster mode)
    container_target = None
    container_target_df = None
    
    if mode != 'solve_cluster':
        avg_upc = calculate_average_upc(containers_df, logger)
        container_target, container_target_df = calculate_container_target(
            wh_id=config['global']['wh_id'],
            planning_timestamp=planning_timestamp,
            active_pickers=labor_headcount,
            avg_pick_uph=config['global']['avg_pick_uph'],
            avg_container_upc=avg_upc,
            variability_factor=config['global']['container_target_variability_factor'],
            containers_df=containers_df,
            tf_min_containers_in_backlog=config['global']['tf_min_containers_in_backlog'],
            logger=logger
        )
    else:
        logger.info("Skipping container target calculation for solve_cluster mode")
    
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
    
    # 4. Write TF preprocessing outputs (container slack and target) to output directory
    # Only write if we have the data and we're not in solve_cluster mode
    if mode != 'solve_cluster':
        # Import the function here to avoid circular imports
        from .data_exchange import write_tf_preprocessing_outputs
        
        # Prepare container slack DataFrame - extract only container-level slack data
        container_slack_df = None
        if containers_df_with_slack is not None and 'slack_minutes' in containers_df_with_slack.columns:
            # Create container slack DataFrame with required fields for database schema
            slack_columns = ['container_id', 'time_until_pull', 'waiting_time', 'picking_time', 
                           'travel_time', 'break_impact', 'other_time', 'total_processing_time', 'slack_minutes', 'slack_category']
            available_columns = [col for col in slack_columns if col in containers_df_with_slack.columns]
            
            if available_columns:
                container_slack_df = containers_df_with_slack[available_columns].drop_duplicates(subset=['container_id'])
                # Add any missing columns with default values
                if 'time_until_pull' not in container_slack_df.columns:
                    container_slack_df['time_until_pull'] = 0.0
                if 'waiting_time' not in container_slack_df.columns:
                    container_slack_df['waiting_time'] = container_slack_df.get('virtual_waiting_time')
                # Add pack_time (same as other_time) with default OTHER_TIME_BUFFER value (usually 5 minutes)
                if 'pack_time' not in container_slack_df.columns:
                    # Use other_time if available, otherwise default to 5.0
                    container_slack_df['pack_time'] = container_slack_df.get('other_time')
                # Add virtual_waiting_time for database compatibility
                if 'virtual_waiting_time' not in container_slack_df.columns:
                    container_slack_df['virtual_waiting_time'] = container_slack_df.get('waiting_time')
                
                # Add wh_id and planning_datetime columns for database consistency
                container_slack_df['wh_id'] = config['global']['wh_id']
                container_slack_df['planning_datetime'] = planning_timestamp
                
                # Reorder columns to match database schema (pack_time = other_time)
                schema_columns = ['wh_id', 'planning_datetime', 'container_id', 'time_until_pull', 'virtual_waiting_time', 'picking_time', 
                                'travel_time', 'pack_time', 'break_impact', 'total_processing_time', 
                                'slack_minutes', 'slack_category']
                # Only include columns that exist in the DataFrame
                ordered_columns = [col for col in schema_columns if col in container_slack_df.columns]
                container_slack_df = container_slack_df[ordered_columns]
        
        # Write preprocessing outputs
        write_tf_preprocessing_outputs(
            output_dir=output_dir,
            container_slack_df=container_slack_df,
            tf_container_target_df=container_target_df,
            logger=logger
        )

    # --- Data Validation Step ---
    # Optimize validation for simulation: skip in solve_cluster mode, load pre-validated data
    if mode == 'solve_cluster':
        # Load pre-validated data for solve_cluster mode
        logger.info("Loading pre-validated data for solve_cluster mode")
        validated_containers_df, validated_skus_df = load_validated_data(working_dir, logger)
        
        if validated_containers_df is not None and validated_skus_df is not None:
            containers_df_with_slack = validated_containers_df
            skus_df = validated_skus_df
            logger.info("Successfully loaded pre-validated data, skipping validation")
        else:
            # Fallback to full validation if pre-validated data not available
            logger.warning("Pre-validated data not available, performing full validation")
            validator = DataValidator(logger=logger)
            try:
                containers_df_with_slack, skus_df = validator.validate(
                    containers_df_with_slack, 
                    skus_df
                )
                logger.info("Data validation successful.")
            except Exception as e:
                logger.error(f"Data validation failed: {e}", exc_info=True)
                raise
    else:
        # Perform full validation for generate_clusters and run_complete modes
        logger.info("Performing data validation")
        validator = DataValidator(logger=logger)
        try:
            containers_df_with_slack, skus_df = validator.validate(
                containers_df_with_slack, 
                skus_df
            )
            logger.info("Data validation successful.")
            
            # Save validated data for future solve_cluster operations
            save_validated_data(working_dir, containers_df_with_slack, skus_df, logger)
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}", exc_info=True)
            raise

    # --- Initialize other components ---
    # Pass output_dir to ContainerClusterer initialization
    # For solve_cluster mode, container_target is not needed since clustering is already done
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
        'working_dir': working_dir,
        'container_target_df': container_target_df
    }


def _perform_clustering(
    clusterer: ContainerClusterer,
    containers_df: pd.DataFrame,
    skus_df: pd.DataFrame, 
    logger: logging.Logger,
    output_dir: str = None,
    wh_id: str = None,
    planning_datetime: datetime = None
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """Perform container clustering and optionally write clustering outputs."""
    try:
        logger.info("Performing container clustering")
      
        clusters, cluster_tours, container_clustering_df, clustering_metadata_df = clusterer.cluster_containers(
            container_data=containers_df,
            slotbook_data=skus_df,
            wh_id=wh_id,
            planning_datetime=planning_datetime
        )
        logger.info(f"Clustering complete. Found {len(clusters)} clusters.")
        
        # --- Create Cluster Metadata --- 
        # The clusterer itself likely holds metadata or can generate it.
        # We need metadata (like tour_id_offset) for the solver service.
        # Assuming the clusterer class has a method or property for this.

        # TODO: Refine how cluster_metadata (esp. tour_id_offset) is obtained
        cluster_metadata = clusterer.cluster_metadata if hasattr(clusterer, 'cluster_metadata') else {}

        # Write clustering outputs to files if output_dir and parameters are provided  
        if output_dir and wh_id and planning_datetime and not container_clustering_df.empty and not clustering_metadata_df.empty:
            try:
                from .data_exchange import write_tf_clustering_outputs
                
                # Write the clustering outputs using the DataFrames returned from clustering
                success = write_tf_clustering_outputs(
                    output_dir=output_dir,
                    container_clustering_df=container_clustering_df,
                    clustering_metadata_df=clustering_metadata_df,
                    logger=logger
                )
                
                if success:
                    logger.info("Successfully wrote clustering output files")
                else:
                    logger.warning("Failed to write clustering output files")
                    
            except Exception as e:
                logger.error(f"Error writing clustering outputs: {e}")
                # Don't fail the overall clustering process

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
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Run the full local workflow: clustering + solving all clusters.
    
    Returns:
        Tuple containing:
            - List of result dictionaries for each cluster
            - Container target DataFrame with metrics
    """
    logger.info("Running local workflow (clustering + solving)")
    clusterer = components['clusterer']
    solver_service: TourFormationSolverService = components['solver_service']
    containers_df = components['containers_df'] # Use potentially updated df
    skus_df = components['skus_df']
    container_target_df = components['container_target_df']

    clusters, cluster_metadata = _perform_clustering(
        clusterer=clusterer,
        containers_df=containers_df,
        skus_df=skus_df,
        logger=logger,
        output_dir=components.get('output_dir'),
        wh_id=config.get('global', {}).get('wh_id'),
        planning_datetime=planning_timestamp
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
    return results, container_target_df


def run_clustering_step(
    components: Dict[str, Any],
    logger: logging.Logger,
    config: Dict[str, Any] = None,
    planning_timestamp: datetime = None
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Run only the clustering step.
    
    Returns:
        Tuple containing:
            - Dictionary mapping cluster IDs to container IDs
            - Dictionary of cluster metadata
            - Container target DataFrame with metrics
    """
    
    logger.info("Running clustering step")
    clusterer = components['clusterer']
    containers_df = components['containers_df']
    skus_df = components['skus_df']
    container_target_df = components['container_target_df']

    
    clusters, cluster_metadata = _perform_clustering(
        clusterer=clusterer,
        containers_df=containers_df,
        skus_df=skus_df,
        logger=logger,
        output_dir=components.get('output_dir'),
        wh_id=config.get('global', {}).get('wh_id') if config else None,
        planning_datetime=planning_timestamp
    )
    
    return clusters, cluster_metadata, container_target_df


def solve_single_cluster(
    components: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    planning_timestamp: datetime,
    cluster_id: str,
    container_ids: List[str],
    cluster_metadata: Dict[str, Any],
    labor_headcount: Optional[int] = None
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Solves a single, specified cluster using the Solver Service.
    
    Returns:
        Tuple containing:
            - Result dictionary for the solved cluster
            - Container target DataFrame with metrics (None for solve_cluster mode)
    """
    
    logger.info(f"Orchestrating solve for single cluster: {cluster_id}")
    solver_service: TourFormationSolverService = components['solver_service']
    containers_df = components['containers_df']
    skus_df = components['skus_df']
    container_target_df = components.get('container_target_df')  # May be None in solve_cluster mode

    # Call the solver service
    result = solver_service.solve_one_cluster(
        container_ids=container_ids,
        containers_df=containers_df,
        skus_df=skus_df,
        planning_timestamp=planning_timestamp,
        cluster_id=cluster_id,
        cluster_metadata=cluster_metadata,
        wh_id = config['global']['wh_id']
    )
    return result, container_target_df


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
        - Tuple[List[Dict], pd.DataFrame]: Results and container target DataFrame for 'run_complete' mode.
        - Tuple[Dict, Dict, pd.DataFrame]: Clusters, metadata, and container target DataFrame for 'generate_clusters' mode.
        - Tuple[Dict, pd.DataFrame]: Result and container target DataFrame for a single cluster in 'solve_cluster' mode.
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

    # Check if TF should be triggered (only applies to run_complete and generate_clusters modes)
    container_target_df = components.get('container_target_df')
    if mode in ['run_complete', 'generate_clusters'] and not should_trigger_tf(container_target_df, logger):
        logger.info("TF processing skipped due to insufficient container backlog")
        # Return appropriate structure based on mode with empty results
        if mode == 'run_complete':
            return [], container_target_df  # Empty results list
        elif mode == 'generate_clusters':
            return {}, {}, container_target_df  # Empty clusters and metadata
    
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
            logger=logger,
            config=config,
            planning_timestamp=planning_timestamp
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
