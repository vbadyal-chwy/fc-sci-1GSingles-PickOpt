"""
Test script for running tour formation via the entrypoint.
Different execution modes - run_complete, generate_clusters, or solve_cluster.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

# Import the tour formation entrypoint
from tour_formation.tf_entry import run_tour_formation_entrypoint

# --- Configuration ---
FC_ID = 'AVP1'                                    # Define FC ID (should come from Platform configuration)
PLANNING_TIMESTAMP_STR = '2025-03-06 14:00:00'    # Define planning timestamp (should come from FlexSim Simulation)
LABOR_HEADCOUNT = 50                              # Define labor headcount (should come from FlexSim Simulation)
MODE = 'generate_clusters'                        # Set the mode: 'run_complete', 'generate_clusters', or 'solve_cluster' 
TARGET_CLUSTER_ID = 1                             # Set the target cluster ID for 'solve_cluster' mode

# Right now, the input and output directories are based on project root.
# INPUT_DIR = 
# OUTPUT_DIR = 
# WORKING_DIR = 
# ---------------------

def main() -> None:
    """Main function to run tour formation entrypoint based on MODE."""
    logger = setup_logging()

    try:
        logger.info(
            f"=== Running Tour Formation in {MODE.upper()} mode ==="
            f"\n=== FC: {FC_ID} ==="
            f"\n=== Planning Timestamp: {PLANNING_TIMESTAMP_STR} ==="
        )

        project_root = Path(__file__).parent
        planning_timestamp = datetime.strptime(
            PLANNING_TIMESTAMP_STR, 
            '%Y-%m-%d %H:%M:%S'
        )
        
        # Setup directories
        input_dir, output_dir, working_dir = setup_directories(
            project_root=project_root,
            fc_id=FC_ID,
            planning_timestamp=planning_timestamp
        )

        cluster_id = TARGET_CLUSTER_ID if MODE == 'solve_cluster' else None
        
        logger.debug(f"Calling run_tour_formation_entrypoint() for mode: {MODE}")
        
        # Call the entrypoint using the final paths
        result = run_tour_formation_entrypoint(
            mode=MODE,
            fc_id=FC_ID,
            planning_timestamp=planning_timestamp,
            input_dir=input_dir,
            output_dir=output_dir,
            working_dir=working_dir,
            labor_headcount=LABOR_HEADCOUNT,
            cluster_id=cluster_id
           
        )
        
        if MODE == 'run_complete':
            logger.info(
                "Check output_dir for final tour results "
            )
        elif MODE == 'generate_clusters':
            logger.info(
                "Check working_dir for subproblem files "
            )
        elif MODE == 'solve_cluster':
            logger.info(
                "Check output_dir for the result file of cluster "
            )

        logger.info(
            f"=== Test for {MODE.upper()} mode complete ==="
        )

    except Exception as e:
        logger.error(
            f"Error running tour formation test in {MODE} mode: {str(e)}", 
            exc_info=True
        )


# --- Utility Function - Setup Directories ---
# This function would be modified for S3 support in Platform.
def setup_directories(
    project_root: Path,
    fc_id: str,
    planning_timestamp: datetime
) -> Tuple[str, str, str]:
    """
    Set up and return the input, output, and working directories.
    """
    # Define base directories
    base_input_dir = project_root / "input"
    base_output_dir = project_root / "output"
    base_working_dir = project_root / "working"
    
    # Construct FC-specific & Planning Timestamp specific paths
    timestamp_dir = planning_timestamp.strftime('%Y%m%d_%H%M%S')
    input_dir = base_input_dir / fc_id / timestamp_dir
    output_dir = base_output_dir / fc_id / timestamp_dir
    working_dir = base_working_dir / fc_id / timestamp_dir

    # Ensure directories exist
    ensure_dir_exists(input_dir)
    ensure_dir_exists(output_dir)
    ensure_dir_exists(working_dir)
    
    return str(input_dir), str(output_dir), str(working_dir)


# --- Utility Function - Ensure Directory Exists ---
def ensure_dir_exists(dir_path: Union[str, Path]) -> str:
    """
    Ensure directory exists, create if it doesn't.
    This function would need to be modified for S3 support.
    """
    path = Path(dir_path) 
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
# -----------------------------------------------------

# --- Utility Function - Logger ---
def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

if __name__ == "__main__":
    main() 