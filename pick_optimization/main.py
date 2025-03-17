"""
Main entry point for Pick Optimization Model.

"""

import os
import sys
from pathlib import Path

# Import Modules
from data.data_puller import DataPuller
from config import load_config
from utils.logging_config import setup_logging
from data.data_validator import DataValidator
from engine.sim_engine import SimEngine

def main():
    """Main execution function."""
    # Determine main directory and config path
    
    main_dir = Path(__file__).parent
    config_path = os.path.join(main_dir, 'config/config.yaml')
    
    # Load configuration
    config = load_config(config_path)
    
    # Set up logging
    logger = setup_logging(config)
    
    try:
        logger.info("Starting Pick Planning Model execution")
        
        # Get input mode and parameters
        input_mode = config['global']['input_mode']
        fc = config['global']['wh_id']
        start_time = config['global']['start_time']
        end_time = config['global']['end_time']
        
        logger.info(f"Running for FC: {fc}")
        logger.info(f"Time window: {start_time} to {end_time}")
        
        # Initialize data puller
        data_puller = DataPuller(config_path, input_mode)
        
        # Pull required data
        logger.info("Fetching input data...")
        container_data = data_puller.get_container_data(fc, start_time, end_time)
        slotbook_data = data_puller.get_slotbook_data(fc)
        
        # Log data statistics
        logger.info(f"Retrieved {len(container_data)} container records")
        logger.info(f"Retrieved {len(slotbook_data)} slotbook records")
        
        # Validate input data
        validator = DataValidator(config)
        container_data, slotbook_data = validator.validate(container_data, slotbook_data)
        
        # Initialize and run simulation engine
        logger.info("Initializing simulation engine...")
        engine = SimEngine(config)
        
        results = engine.run(
            container_data=container_data,
            slotbook_data=slotbook_data,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info("Simulation completed successfully")
        
        # Log summary statistics from results
        log_summary_statistics(results, logger)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in Pick Planning Model execution: {str(e)}")
        logger.exception("Stack trace:")
        return 1

def log_summary_statistics(results, logger):
    """Log summary statistics from simulation results."""
    # Get consolidated results
    consolidated = results.get_consolidated_results()
    
    # Log formation stats
    if consolidated['formation_stats'] is not None:
        formation_stats = consolidated['formation_stats']
        total_tours = formation_stats['TotalTours'].sum()
        total_units = formation_stats['TotalUnits'].sum()
        logger.info(f"Total tours formed: {total_tours}")
        logger.info(f"Total units processed: {total_units}")
    
    # Log allocation stats
    if consolidated['allocation_stats'] is not None:
        allocation_stats = consolidated['allocation_stats']
        if 'ToursReleased' in allocation_stats.columns:
            total_released = allocation_stats['ToursReleased'].sum()
            logger.info(f"Total tours released: {total_released}")
    
    # Log buffer assignments
    if consolidated['buffer_assignments'] is not None:
        buffer_assignments = consolidated['buffer_assignments']
        total_assignments = len(buffer_assignments)
        unique_buffers = buffer_assignments['BufferSpotID'].nunique()
        logger.info(f"Tours assigned to buffer spots: {total_assignments}")
        logger.info(f"Unique buffer spots used: {unique_buffers}")

if __name__ == "__main__":
    sys.exit(main())