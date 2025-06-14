"""
Main orchestration script for external data loading.

This script coordinates the full pipeline:
1. Extract data from Snowflake
2. Transform data to match SQLite schema
3. Import data into SQLite database
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from snowflake_extractor import ExternalDataExtractor
from data_transformer import DataTransformer
from sqlite_importer import SQLiteDataImporter


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('external_data_loading.log')
        ]
    )


def generate_execution_id() -> str:
    """Generate a unique execution ID based on timestamp."""
    return f"ext_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_external_data(config_path: str, db_path: str, 
                      execution_id: Optional[str] = None) -> bool:
    """
    Main function to load external data from Snowflake to SQLite.
    
    Parameters
    ----------
    config_path : str
        Path to sim_config.yaml
    db_path : str
        Path to SQLite database
    execution_id : str, optional
        Custom execution ID, generates one if not provided
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting external data loading pipeline")
    
    # Generate execution ID if not provided
    if not execution_id:
        execution_id = generate_execution_id()
    
    logger.info(f"Execution ID: {execution_id}")
    
    try:
        # Step 1: Initialize components
        logger.info("Initializing data extraction components")
        
        extractor = ExternalDataExtractor(config_path)
        transformer = DataTransformer(execution_id)
        importer = SQLiteDataImporter(db_path)
        
        # Get configuration parameters
        config = extractor.config
        global_config = config.get('global', {})
        wh_id = global_config.get('wh_id')
        start_time = global_config.get('start_time')
        end_time = global_config.get('end_time')
        
        if not all([wh_id, start_time, end_time]):
            logger.error("Missing required configuration: wh_id, start_time, or end_time")
            return False
        
        logger.info(f"Configuration: wh_id={wh_id}, start_time={start_time}, end_time={end_time}")
        
        # Step 2: Extract data from Snowflake
        logger.info("Starting data extraction from Snowflake")
        raw_data = extractor.extract_all_data()
        
        # Log extraction results
        for key, df in raw_data.items():
            logger.info(f"Extracted {key}: {len(df)} rows")
        
        # Step 3: Transform data
        logger.info("Starting data transformation")
        transformed_data = transformer.transform_all_data(raw_data, wh_id)
        
        # Validate transformed data
        if not transformer.validate_transformed_data(transformed_data):
            logger.error("Data validation failed, aborting import")
            return False
        
        # Step 4: Import data into SQLite
        logger.info("Starting data import to SQLite")
        import_success = importer.import_all_data(
            transformed_data, execution_id, wh_id, start_time, end_time
        )
        
        if import_success:
            # Get and log import summary
            summary = importer.get_import_summary(execution_id)
            logger.info(f"Import completed successfully. Summary: {summary}")
            return True
        else:
            logger.error("Data import failed")
            return False
    
    except Exception as e:
        logger.error(f"External data loading pipeline failed: {e}")
        return False


def main():
    """Command line interface for external data loading."""
    parser = argparse.ArgumentParser(
        description="Load external data from Snowflake to SQLite"
    )
    
    parser.add_argument(
        '--config', 
        required=True,
        help="Path to sim_config.yaml file"
    )
    
    parser.add_argument(
        '--database', 
        required=True,
        help="Path to SQLite database file"
    )
    
    parser.add_argument(
        '--execution-id',
        help="Custom execution ID (optional, will generate if not provided)"
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("External data loading script started")
    
    # Validate input files
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    db_path = Path(args.database)
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
    
    # Run the pipeline
    success = load_external_data(
        str(config_path), 
        str(db_path), 
        args.execution_id
    )
    
    if success:
        logger.info("External data loading completed successfully")
        sys.exit(0)
    else:
        logger.error("External data loading failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 