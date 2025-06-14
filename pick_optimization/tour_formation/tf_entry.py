"""
Container entry point for tour formation.

Handles input/output via flat files, and invokes the
tour formation orchestrator based on the execution mode.
"""

from datetime import datetime
from typing import Optional, Union
from pathlib import Path

from .tf_orchestrator import orchestrate_tour_formation
from .data_exchange import (
    load_container_data,    
    load_slotbook_data,
    read_subproblem,
    write_results,
    write_subproblems,
    normalize_path
)
from .utils import load_model_config
from pick_optimization.utils.logging_config import setup_logging, get_workflow_logger

# Get module-specific logger - will be enhanced with workflow logging
logger = get_workflow_logger(__name__, 'tour_formation')

def run_tour_formation_entrypoint(
    mode: str,
    fc_id: str,
    planning_timestamp: Union[str, datetime],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    working_dir: Union[str, Path],
    labor_headcount: Optional[int],
    cluster_id: Optional[int] = None,
):
    """Main tour formation entry point function.
    
    Args:
        mode: Execution mode ('run_complete', 'generate_clusters', or 'solve_cluster')
        fc_id: Fulfillment center ID
        planning_timestamp: Planning timestamp (string in ISO format or datetime object)
        input_dir: Directory containing input data files
        output_dir: Directory where output files will be written
        working_dir: Directory for intermediate working files
        cluster_id: ID of cluster to solve (required for 'solve_cluster' mode)
        labor_headcount: Number of available labor headcount from FlexSim Simulation
    
    """
    try:
        # Validate and normalize inputs
        input_dir = normalize_path(input_dir)
        output_dir = normalize_path(output_dir)
        working_dir = normalize_path(working_dir)
        
        # Load config & setup centralized logging
        config = load_model_config(input_dir)
        
        # Initialize centralized logging system
        setup_logging(config, 'tour_formation')
        
        # Get workflow-specific logger
        logger = get_workflow_logger(__name__, 'tour_formation')
        
        
        # Validate mode
        valid_modes = ['run_complete', 'generate_clusters', 'solve_cluster']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {valid_modes}")

        # Validate cluster_id
        if mode == 'solve_cluster' and cluster_id is None:
            raise ValueError("cluster_id is required for mode 'solve_cluster'")
        
        # Validate and parse planning_timestamp
        if isinstance(planning_timestamp, str):
            try:
                planning_timestamp = datetime.fromisoformat(planning_timestamp)
            except ValueError as e:
                raise ValueError(
                    f"Invalid planning_timestamp format: {planning_timestamp}. "
                    f"Expected ISO format (e.g., 2024-05-15T14:30:00)"
                ) from e

        logger.info(f"Starting Tour Formation in {mode.upper()} mode.")
        # Log the directories received
        logger.debug(f"Using Input Dir: {input_dir}")
        logger.debug(f"Using Output Dir: {output_dir}") 
        logger.debug(f"Using Working Dir: {working_dir}")
        if labor_headcount is not None:
            logger.debug(f"Labor Headcount: {labor_headcount}")
        if cluster_id is not None:
            logger.debug(f"Target Cluster ID (for solve mode): {cluster_id}")

        # Load Base Inputs (Always needed for all modes) 
        logger.debug("Loading base input data (container_data, slotbook_data)...")
        containers_df = load_container_data(input_dir)
        slotbook_df = load_slotbook_data(input_dir)
        logger.info("Base input data loaded.")

        # Prepare Orchestrator Arguments based on Mode
        orchestrator_kwargs = {
            'containers_df': containers_df,
            'skus_df': slotbook_df,
            'config': config,
            'logger': logger,
            'planning_timestamp': planning_timestamp,
            'mode': mode,  
            'output_dir': output_dir,
            'working_dir': working_dir,
            'labor_headcount': labor_headcount
        }
        
        cluster_to_solve_data = None
        if mode == 'solve_cluster':
            logger.debug(f"Loading subproblem data for: cluster_{cluster_id}")
            cluster_to_solve_data = read_subproblem(
                working_dir=working_dir,
                cluster_id=cluster_id,
                logger=logger
            )
            if not cluster_to_solve_data:
                raise FileNotFoundError(f"Subproblem data not found for cluster_{cluster_id} in {working_dir}")

            # Add specific cluster data to kwargs for the orchestrator
            orchestrator_kwargs['cluster_to_solve_id'] = cluster_id
            orchestrator_kwargs['cluster_to_solve_containers'] = cluster_to_solve_data['container_ids']
            orchestrator_kwargs['cluster_to_solve_metadata'] = cluster_to_solve_data['metadata']
            logger.info(f"Subproblem data for cluster {cluster_id} loaded.")

        # Execute Orchestrator
        logger.debug("Invoking tour formation orchestrator...")
        orchestrator_result = orchestrate_tour_formation(**orchestrator_kwargs)
        logger.debug("Orchestrator execution finished.")

        # Handle Outputs based on mode
        if mode == 'run_complete':
            if orchestrator_result:
                logger.info(f"Writing {len(orchestrator_result)} cluster results to output directory")
                write_results(output_dir, orchestrator_result, logger)
            else:
                logger.warning("Orchestrator returned no results in run_complete mode.")

        elif mode == 'generate_clusters':
            clusters, cluster_metadata, container_target_df = orchestrator_result
            if clusters:
                logger.info(f"Writing {len(clusters)} subproblems to working directory") 
                write_subproblems(
                    working_dir=working_dir,
                    output_dir=output_dir,
                    clusters=clusters,
                    cluster_metadata=cluster_metadata,
                    container_target_df=container_target_df,
                    logger=logger
                )
            else:
                logger.warning("Orchestrator returned no clusters in generate_clusters mode.")

        elif mode == 'solve_cluster':
            if orchestrator_result:
                result_dict, container_target_df = orchestrator_result
                logger.info(f"Writing result for cluster {cluster_id} to output directory") 
                write_results(output_dir, [result_dict], logger)
            else:
                logger.error(f"Orchestrator failed to return a result for cluster {cluster_id} in solve_cluster mode.")

        logger.info("Tour Formation process completed successfully.")
        return orchestrator_result

    except Exception as e:
        print(f"Tour Formation process failed: {str(e)}")
        raise
