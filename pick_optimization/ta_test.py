"""
Test script for running tour allocation via the entrypoint.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple

# Import the tour allocation entry point
from tour_allocation.ta_entry import run_tour_allocation_entrypoint 

# --- Configuration ---
FC_ID = 'AVP1'                                   # Define FC ID
PLANNING_TIMESTAMP_STR = '2025-04-06 14:00:00'   # Define planning timestamp
TARGET_TOURS = 5                                 # Define target number of tours to release

# Right now, the input and output directories are based on project root.
# INPUT_DIR = 
# OUTPUT_DIR = 
# WORKING_DIR = 
# ---------------------

# --- Utility Function - Ensure Directory Exists ---
def ensure_dir_exists(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    Returns the Path object.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# --- Utility Function - Setup Directories ---
def setup_directories(
    project_root: Path,
    fc_id: str,
    planning_timestamp: datetime
) -> Tuple[Path, Path]:
    """
    Set up and return the input and output directories as Path objects.
    """
    # Define base directories relative to the project root 
    base_input_dir = project_root / "input"
    base_output_dir = project_root / "output"

    # Construct FC-specific & Planning Timestamp specific paths
    timestamp_dir = planning_timestamp.strftime('%Y%m%d_%H%M%S')
    
   
    input_dir = base_input_dir / fc_id / timestamp_dir
    output_dir = base_output_dir / fc_id / timestamp_dir

    # Ensure directories exist
    ensure_dir_exists(input_dir)
    ensure_dir_exists(output_dir)

    return input_dir, output_dir

# --- Utility Function - Logger ---
def setup_logging() -> logging.Logger:
    """Set up logging configuration.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main() -> None:
    """Main function to run tour allocation test."""
    logger = setup_logging()

    try:
        logger.info(
            f"=== Running Tour Allocation Test for FC: {FC_ID} ===\n"
            f"=== Planning Timestamp: {PLANNING_TIMESTAMP_STR} ==="
        )

        project_root = Path(__file__).parent 

        planning_timestamp = datetime.strptime(
            PLANNING_TIMESTAMP_STR,
            '%Y-%m-%d %H:%M:%S'
        )


        # Setup directories
        input_dir, output_dir = setup_directories(
            project_root=project_root,
            fc_id=FC_ID,
            planning_timestamp=planning_timestamp
        )

        logger.info("Calling run_tour_allocation_entrypoint()")

        # Call the entrypoint with the required parameters
        run_tour_allocation_entrypoint(
            fc_id=FC_ID,
            planning_timestamp=planning_timestamp,
            input_dir=input_dir,
            output_dir=output_dir,
            target_tours=TARGET_TOURS
        )

        logger.info("run_tour_allocation_entrypoint() completed.")

        logger.info(
            "Check output_dir for final allocation results "
        )

        logger.info(
            "=== Tour Allocation Test complete ==="
        )

    except Exception as e:
        logger.error(
            f"Error running tour allocation test: {str(e)}",
            exc_info=True
        )

if __name__ == "__main__":
    main() 