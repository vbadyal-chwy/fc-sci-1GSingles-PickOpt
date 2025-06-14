"""
End-to-end simulation for Pick Optimization with database-driven approach.
Combines database workflow from simulator.ipynb with structure from end_to_end_test.py.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import yaml

# Add project paths
project_root = Path(__file__).parent.parent  # Go up one more level since we're now in simulator/
sys.path.append(str(project_root))
sys.path.append(str(project_root / "simulator" / "data_store"))

# Database infrastructure imports
from simulator.data_store.create_database import create_database  
from simulator.data_store.external_data.snowflake_extractor import ExternalDataExtractor
from simulator.data_store.external_data.data_transformer import DataTransformer
from simulator.data_store.external_data.sqlite_importer import SQLiteDataImporter
from simulator.data_store.core.db_manager import SimulationDBManager

# Model entrypoints
from pick_optimization.tour_formation.tf_entry import run_tour_formation_entrypoint
from pick_optimization.tour_allocation.ta_entry import run_tour_allocation_entrypoint

# Database input/output processors
from simulator.utils.database_tf_inputs import create_tf_inputs_from_database, validate_tf_input_data
from simulator.utils.database_ta_inputs import create_ta_inputs_from_database, get_ready_tours_summary
from simulator.utils.tf_output_processor import process_tf_outputs_to_database, cleanup_tf_directories
from simulator.utils.ta_output_processor import process_ta_outputs_to_database as process_ta_outputs_to_db, cleanup_ta_directories as cleanup_ta_dirs

# Model trigger logic
from simulator.utils.model_trigger_logic import (
    check_tf_trigger_conditions,
    check_ta_trigger_conditions, 
    record_trigger_decision,
    get_last_successful_tf_timestamp
)


def setup_logging(execution_id: str) -> logging.Logger:
    """Setup logging for simulation."""
    #log_file = f"simulation_{execution_id}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('end_to_end_simulation')
    logger.info("=" * 60)
    logger.info("END-TO-END SIMULATION STARTED")
    logger.info("=" * 60)
    
    return logger


def load_config(config_path: str) -> dict:
    """Load simulation configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_simulation_database(execution_id: str, data_store_path: Path, logger: logging.Logger) -> Path:
    """
    Step 2.1: Create new SQLite database with schema.
    """
    logger.info("Step 2.1: Creating simulation database")
    
    db_name = f"simulation_{execution_id}.db"
    db_path = data_store_path / db_name
    schema_path = data_store_path / "database_schema.sql"
    
    logger.info(f"Creating database: {db_path}")
    
    success = create_database(str(db_path), str(schema_path))
    
    if success:
        logger.info(f"Database created successfully: {db_name}")
        
        # Verify database structure
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Database tables created: {len(tables)}")
        for table in tables:
            logger.info(f"   - {table}")
        
        return db_path
    else:
        raise Exception("Database creation failed")


def load_external_data(config_path: str, db_path: Path, execution_id: str, logger: logging.Logger) -> bool:
    """
    Step 2.2: Extract, transform, and load external data.
    """
    logger.info("Step 2.2: Loading external data")
    
    try:
        # Initialize components
        extractor = ExternalDataExtractor(config_path)
        transformer = DataTransformer(execution_id)
        importer = SQLiteDataImporter(str(db_path))
        
        logger.info("Data pipeline components initialized")
        
        # Extract external data
        logger.info("Extracting data from external sources...")
        raw_data = extractor.extract_all_data()
        
        logger.info("Data extraction completed")
        total_rows = sum(len(df) for df in raw_data.values())
        logger.info(f"Total rows extracted: {total_rows:,}")
        
        # Transform data
        logger.info("Transforming data to database schema...")
        # Load config to get wh_id
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        wh_id = config['global']['wh_id']
        
        transformed_data = transformer.transform_all_data(raw_data, wh_id)
        
        logger.info("Data transformation completed")
        total_transformed_rows = sum(len(df) for df in transformed_data.values())
        logger.info(f"Total transformed rows: {total_transformed_rows:,}")
        
        # Import to database
        logger.info("Importing data to SQLite database...")
        global_config = config['global']
        pick_opt_config = config.get('pick_optimization', {})
        
        import_success = importer.import_all_data(
            transformed_data=transformed_data,
            execution_id=execution_id,
            wh_id=wh_id,
            start_time=global_config['start_time'],
            end_time=global_config['end_time'],
            tf_interval_minutes=pick_opt_config.get('tf_interval_minutes', 30),
            ta_interval_minutes=pick_opt_config.get('ta_interval_minutes', 5),
            tf_min_containers_in_backlog=pick_opt_config.get('tf_min_containers_in_backlog', 1000)
        )
        
        if import_success:
            logger.info("Database import completed successfully")
            
            # Get import summary
            summary = importer.get_import_summary(execution_id)
            if 'error' not in summary:
                logger.info("Import Summary:")
                logger.info(f"   - Execution ID: {summary['execution_id']}")
                logger.info(f"   - Warehouse: {summary['wh_id']}")
                logger.info(f"   - Status: {summary['status']}")
                
                counts = summary['counts']
                total_records = sum(counts.values())
                for table, count in counts.items():
                    logger.info(f"   - {table}: {count:,} records")
                logger.info(f"Total records imported: {total_records:,}")
            
            return True
        else:
            logger.error("Database import failed")
            return False
            
    except Exception as e:
        logger.error(f"External data loading failed: {e}", exc_info=True)
        return False


def validate_database(db_path: Path, execution_id: str, logger: logging.Logger) -> bool:
    """
    Step 2.3: Validate database contents after external data load.
    """
    logger.info("Step 2.3: Validating database contents")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Check execution record
            exec_query = "SELECT * FROM executions WHERE execution_id = ?"
            exec_df = pd.read_sql_query(exec_query, conn, params=[execution_id])
            
            if len(exec_df) == 1:
                logger.info("Execution record found")
                exec_row = exec_df.iloc[0]
                logger.info(f"   - Status: {exec_row['status']}")
                logger.info(f"   - TF Interval: {exec_row['tf_interval_minutes']} minutes")
                logger.info(f"   - TA Interval: {exec_row['ta_interval_minutes']} minutes")
            else:
                logger.error("Execution record not found or duplicated")
                return False
            
            # Check each table
            tables_to_check = ['containers', 'container_details', 'slotbook', 'labor_data']
            
            logger.info("Table validation:")
            for table in tables_to_check:
                count_query = f"SELECT COUNT(*) as count FROM {table} WHERE execution_id = ?"
                result = conn.execute(count_query, [execution_id]).fetchone()
                count = result[0]
                
                if count > 0:
                    logger.info(f"   {table}: {count:,} records")
                else:
                    logger.warning(f"   {table}: No records found")
            
            logger.info("Database validation completed")
            return True
            
    except Exception as e:
        logger.error(f"Database validation failed: {e}", exc_info=True)
        return False


def create_tf_inputs(execution_id: str, planning_timestamp: datetime, config: dict, 
                    db_path: Path, logger: logging.Logger) -> tuple[Path, bool]:
    """
    Step 3.1: Create TF inputs from database.
    """
    logger.info(f"Step 3.1: Creating TF inputs for {planning_timestamp}")
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        
        # Setup paths
        base_input_dir = project_root / "pick_optimization" / "input"
        tf_config_path = project_root / config['paths']['tf_config']
        
        # Create TF inputs from database
        tf_input_dir = create_tf_inputs_from_database(
            execution_id=execution_id,
            planning_timestamp=planning_timestamp,
            db_path=db_path,
            wh_id=wh_id,
            base_input_dir=base_input_dir,
            tour_formation_config_path=tf_config_path,
            logger=logger
        )
        
        # Validate inputs
        validation_passed = validate_tf_input_data(tf_input_dir, logger)
        
        if validation_passed:
            logger.info(f"TF inputs created successfully: {tf_input_dir}")
            return tf_input_dir, True
        else:
            logger.error("TF input validation failed")
            return None, False
            
    except Exception as e:
        logger.error(f"TF input creation failed: {e}", exc_info=True)
        return None, False


def run_tf_model(planning_timestamp: datetime, tf_input_dir: Path, config: dict, 
                logger: logging.Logger) -> dict:
    """
    Step 3.2: Run TF model (generate_clusters + solve_clusters).
    """
    logger.info(f"Step 3.2: Running TF model for {planning_timestamp}")
    
    result = {
        'success': False,
        'output_dir': None,
        'clusters_solved': 0,
        'total_clusters': 0,
        'duration_seconds': 0
    }
    
    start_time = datetime.now()
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        
        # Setup output directories
        timestamp_str = planning_timestamp.strftime('%Y%m%d_%H%M%S')
        output_dir = project_root / "pick_optimization" / "output" / wh_id / timestamp_str
        working_dir = project_root / "pick_optimization" / "working" / wh_id
        
        # Ensure directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        result['output_dir'] = output_dir
        
        # Step 3.2.1: Generate clusters
        logger.info("Running TF generate_clusters...")
        run_tour_formation_entrypoint(
            mode='generate_clusters',
            fc_id=wh_id,
            planning_timestamp=planning_timestamp,
            input_dir=str(tf_input_dir),
            output_dir=str(output_dir),
            working_dir=str(working_dir),
            labor_headcount=50  # Default value
        )
        logger.info("TF generate_clusters completed")
        
        # Step 3.2.2: Solve clusters
        logger.info("Running TF solve_clusters...")
        
        # Read metadata to find cluster IDs
        metadata_path = output_dir / "clustering_metadata.csv"
        if metadata_path.is_file():
            metadata_df = pd.read_csv(metadata_path)
            if 'cluster_id' in metadata_df.columns:
                cluster_ids = metadata_df['cluster_id'].unique().tolist()
                result['total_clusters'] = len(cluster_ids)
                logger.info(f"Found {len(cluster_ids)} clusters to solve: {cluster_ids}")
                
                clusters_solved = 0
                for cluster_id in cluster_ids:
                    logger.info(f"Solving Cluster ID: {cluster_id}")
                    
                    run_tour_formation_entrypoint(
                        mode='solve_cluster',
                        fc_id=wh_id,
                        planning_timestamp=planning_timestamp,
                        input_dir=str(tf_input_dir),
                        output_dir=str(output_dir),
                        working_dir=str(working_dir),
                        labor_headcount=50,
                        cluster_id=int(cluster_id)
                    )
                    
                    clusters_solved += 1
                    logger.info(f"Cluster {cluster_id} solved successfully")
                
                result['clusters_solved'] = clusters_solved
                
                if clusters_solved == len(cluster_ids):
                    result['success'] = True
                    logger.info(f"All {clusters_solved} clusters solved successfully")
                else:
                    logger.error(f"Only {clusters_solved}/{len(cluster_ids)} clusters solved")
            else:
                logger.error("No 'cluster_id' column found in metadata")
        else:
            logger.error(f"Metadata file not found: {metadata_path}")
        
        duration = (datetime.now() - start_time).total_seconds()
        result['duration_seconds'] = duration
        logger.info(f"TF model completed in {duration:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"TF model execution failed: {e}", exc_info=True)
        result['error'] = str(e)
        return result


def process_tf_outputs(execution_id: str, planning_timestamp: datetime, tf_output_dir: Path,
                      db_path: Path, config: dict, logger: logging.Logger) -> bool:
    """
    Step 3.3: Process TF outputs to database and cleanup files.
    """
    logger.info(f"Step 3.3: Processing TF outputs for {planning_timestamp}")
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        
        # Initialize database manager
        db_manager = SimulationDBManager(str(db_path), execution_id)
        
        # Process outputs to database
        logger.info("Processing TF outputs to database...")
        success = process_tf_outputs_to_database(
            output_dir=tf_output_dir,
            db_manager=db_manager,
            wh_id=wh_id,
            planning_datetime=planning_timestamp,
            logger=logger
        )
        
        if success:
            logger.info("TF outputs processed to database successfully")
            
            # Cleanup directories
            logger.info("Cleaning up TF directories...")
            cleanup_success = cleanup_tf_directories(
                base_input_dir=project_root / "pick_optimization" / "input",
                base_output_dir=project_root / "pick_optimization" / "output",
                base_working_dir=project_root / "pick_optimization" / "working",
                wh_id=wh_id,
                logger=logger
            )
            
            if cleanup_success:
                logger.info("TF directory cleanup completed")
            else:
                logger.warning("TF directory cleanup had issues")
            
            return True
        else:
            logger.error("TF output processing failed")
            return False
            
    except Exception as e:
        logger.error(f"TF output processing failed: {e}", exc_info=True)
        return False


def create_ta_inputs(execution_id: str, planning_timestamp: datetime, config: dict,
                    db_path: Path, logger: logging.Logger) -> tuple[Path, dict, bool]:
    """
    Step 4.1: Create TA inputs from database.
    """
    logger.info(f"Step 4.1: Creating TA inputs for {planning_timestamp}")
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        
        # Check ready tours availability
        ready_tours_summary = get_ready_tours_summary(
            execution_id=execution_id,
            db_path=db_path,
            wh_id=wh_id,
            logger=logger
        )
        
        logger.info(f"Ready tours summary: {ready_tours_summary}")
        
        if ready_tours_summary['tour_count'] == 0:
            logger.warning("No ready tours available for TA")
            return None, ready_tours_summary, False
        
        # Setup paths
        base_input_dir = project_root / "pick_optimization" / "input"
        ta_config_path = project_root / config['paths']['ta_config']
        data_dir = project_root / "simulator" / "data"
        
        # Create TA inputs from database
        ta_input_dir = create_ta_inputs_from_database(
            execution_id=execution_id,
            planning_timestamp=planning_timestamp,
            db_path=db_path,
            wh_id=wh_id,
            base_input_dir=base_input_dir,
            tour_allocation_config_path=ta_config_path,
            data_dir=data_dir,
            logger=logger
        )
        
        logger.info(f"TA inputs created successfully: {ta_input_dir}")
        return ta_input_dir, ready_tours_summary, True
        
    except Exception as e:
        logger.error(f"TA input creation failed: {e}", exc_info=True)
        return None, {}, False


def run_ta_model(planning_timestamp: datetime, ta_input_dir: Path, ready_tours_summary: dict,
                config: dict, logger: logging.Logger) -> dict:
    """
    Step 4.2: Run TA model.
    """
    logger.info(f"Step 4.2: Running TA model for {planning_timestamp}")
    
    result = {
        'success': False,
        'output_dir': None,
        'target_tours': 0,
        'duration_seconds': 0
    }
    
    start_time = datetime.now()
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        
        # Setup output directory
        timestamp_str = planning_timestamp.strftime('%Y%m%d_%H%M%S')
        output_dir = project_root / "pick_optimization" / "output" / wh_id / timestamp_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result['output_dir'] = output_dir
        
        # Calculate target tours
        pick_opt_config = config.get('pick_optimization', {})
        target_tours_per_iteration = pick_opt_config.get('target_tours_per_iteration', 5)
        target_tours = min(target_tours_per_iteration, ready_tours_summary['tour_count'])
        result['target_tours'] = target_tours
        
        logger.info(f"Target tours for this iteration: {target_tours}")
        
        # Run TA model
        logger.info("Running TA model...")
        run_tour_allocation_entrypoint(
            fc_id=wh_id,
            planning_timestamp=planning_timestamp,
            input_dir=str(ta_input_dir),
            output_dir=str(output_dir),
            target_tours=target_tours
        )
        
        result['success'] = True
        duration = (datetime.now() - start_time).total_seconds()
        result['duration_seconds'] = duration
        logger.info(f"TA model completed in {duration:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"TA model execution failed: {e}", exc_info=True)
        result['error'] = str(e)
        return result


def process_ta_outputs(execution_id: str, planning_timestamp: datetime, ta_output_dir: Path,
                      db_path: Path, config: dict, logger: logging.Logger) -> bool:
    """
    Step 4.3: Process TA outputs to database and cleanup files.
    """
    logger.info(f"Step 4.3: Processing TA outputs for {planning_timestamp}")
    
    try:
        global_config = config['global']
        wh_id = global_config['wh_id']
        timestamp_str = planning_timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Process outputs to database using TA output processor
        logger.info("Processing TA outputs to database...")
        
        # Get start_time from config for FlexSim time calculations
        start_time = datetime.strptime(config['global']['start_time'], '%Y-%m-%d %H:%M:%S')
        config_path = project_root / "simulator" / "config" / "sim_config.yaml"
        
        process_ta_outputs_to_db(
            execution_id=execution_id,
            wh_id=wh_id,
            planning_datetime=planning_timestamp,
            output_dir=ta_output_dir,
            db_path=db_path,
            config_path=config_path,
            start_time=start_time,
            logger=logger
        )
        
        logger.info("TA outputs processed to database successfully")
        
        # Cleanup directories using TA output processor
        logger.info("Cleaning up TA directories...")
        input_dir = project_root / "pick_optimization" / "input" / wh_id / timestamp_str
        cleanup_ta_dirs(
            input_dir=input_dir if input_dir.exists() else None,
            output_dir=ta_output_dir,
            logger=logger
        )
        
        logger.info("TA directory cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"TA output processing failed: {e}", exc_info=True)
        return False


def run_simulation(config: dict, db_path: Path, execution_id: str, logger: logging.Logger) -> dict:
    """
    Step 5.1: Main simulation loop with time-based TF/TA scheduling.
    """
    logger.info("Step 5.1: Starting main simulation loop")
    
    simulation_results = {
        'execution_id': execution_id,
        'tf_results': [],
        'ta_results': [],
        'total_iterations': 0,
        'successful_iterations': 0,
        'failed_iterations': 0
    }
    
    try:
        # Parse time configuration
        global_config = config['global']
        pick_opt_config = config.get('pick_optimization', {})
        
        start_time = datetime.strptime(global_config['start_time'], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(global_config['end_time'], '%Y-%m-%d %H:%M:%S')
        
        tf_interval = timedelta(minutes=pick_opt_config.get('tf_interval_minutes', 30))
        ta_interval = timedelta(minutes=pick_opt_config.get('ta_interval_minutes', 5))
        
        logger.info(f"Simulation time range: {start_time} to {end_time}")
        logger.info(f"TF interval: {tf_interval}, TA interval: {ta_interval}")
        
        # Time management with enhanced trigger logic
        current_time = start_time
        next_ta_time = start_time
        iteration = 1
        total_iterations = 10
        last_tf_time = None
        
        # Main simulation loop
        while current_time <= end_time and iteration <= total_iterations:
            logger.info(f"=== Simulation Time: {current_time} ===")
            
            is_first_iteration = (iteration == 1)
            
            # Check TF trigger conditions
            should_run_tf, tf_reason, tf_metadata = check_tf_trigger_conditions(
                execution_id=execution_id,
                db_path=db_path,
                wh_id=config['global']['wh_id'],
                current_time=current_time,
                config=config,
                last_tf_time=last_tf_time,
                is_first_iteration=is_first_iteration
            )
            
            # Record TF trigger decision
            record_trigger_decision(
                db_path=db_path,
                execution_id=execution_id,
                planning_datetime=current_time,
                model_type='TF',
                triggered=should_run_tf,
                reason=tf_reason,
                metadata=tf_metadata
            )
            
            # Execute TF iteration
            if should_run_tf:
                logger.info(f"*** Running Tour Formation at {current_time}: {tf_reason} ***")
                logger.info(f"    Backlog: {tf_metadata['backlog_container_count']}/{tf_metadata['tf_min_containers_threshold']}")
                logger.info(f"    Ready tours: {tf_metadata['ready_tours_count']}")
                
                # Step 3.1: Create TF inputs
                tf_input_dir, tf_input_success = create_tf_inputs(
                    execution_id, current_time, config, db_path, logger
                )
                
                if not tf_input_success:
                    logger.error("TF input creation failed - stopping simulation")
                    simulation_results['failed_iterations'] += 1
                    break
                
                # Step 3.2: Run TF model
                tf_result = run_tf_model(current_time, tf_input_dir, config, logger)
                
                if not tf_result['success']:
                    logger.error("TF model execution failed - stopping simulation")
                    simulation_results['tf_results'].append(tf_result)
                    simulation_results['failed_iterations'] += 1
                    break
                
                # Step 3.3: Process TF outputs
                tf_output_success = process_tf_outputs(
                    execution_id, current_time, tf_result['output_dir'], db_path, config, logger
                )
                
                if not tf_output_success:
                    logger.error("TF output processing failed - stopping simulation")
                    simulation_results['tf_results'].append(tf_result)
                    simulation_results['failed_iterations'] += 1
                    break
                
                simulation_results['tf_results'].append(tf_result)
                simulation_results['successful_iterations'] += 1
                last_tf_time = current_time  # Update last successful TF time
                logger.info(f"*** Tour Formation completed successfully for {current_time} ***")
            else:
                logger.info(f"*** Skipping Tour Formation at {current_time}: {tf_reason} ***")
            
            # Check TA trigger conditions (every 5 minutes - time-based evaluation)
            if current_time >= next_ta_time:
                should_run_ta, ta_reason, ta_metadata = check_ta_trigger_conditions(
                    execution_id=execution_id,
                    db_path=db_path,
                    wh_id=config['global']['wh_id'],
                    config=config
                )
                
                # Record TA trigger decision
                record_trigger_decision(
                    db_path=db_path,
                    execution_id=execution_id,
                    planning_datetime=current_time,
                    model_type='TA',
                    triggered=should_run_ta,
                    reason=ta_reason,
                    metadata=ta_metadata
                )
                
                # Execute TA iteration
                if should_run_ta:
                    logger.info(f"--- Running Tour Allocation at {current_time}: {ta_reason} ---")
                    logger.info(f"    Ready tours: {ta_metadata['ready_tours_count']}")
                    
                    # Step 4.1: Create TA inputs
                    ta_input_dir, ready_tours_summary, ta_input_success = create_ta_inputs(
                        execution_id, current_time, config, db_path, logger
                    )
                    
                    if not ta_input_success:
                        logger.error("TA input creation failed - stopping simulation")
                        simulation_results['failed_iterations'] += 1
                        break
                    
                    # Step 4.2: Run TA model
                    ta_result = run_ta_model(current_time, ta_input_dir, ready_tours_summary, config, logger)
                    
                    if not ta_result['success']:
                        logger.error("TA model execution failed - stopping simulation")
                        simulation_results['ta_results'].append(ta_result)
                        simulation_results['failed_iterations'] += 1
                        break
                    
                    # Step 4.3: Process TA outputs
                    ta_output_success = process_ta_outputs(
                        execution_id, current_time, ta_result['output_dir'], db_path, config, logger
                    )
                    
                    if not ta_output_success:
                        logger.error("TA output processing failed - stopping simulation")
                        simulation_results['ta_results'].append(ta_result)
                        simulation_results['failed_iterations'] += 1
                        break
                    
                    simulation_results['ta_results'].append(ta_result)
                    simulation_results['successful_iterations'] += 1
                    logger.info(f"--- Tour Allocation completed successfully for {current_time} ---")
                else:
                    logger.info(f"--- Skipping Tour Allocation at {current_time}: {ta_reason} ---")
                
                next_ta_time += ta_interval
            
            # Advance simulation time
            if next_ta_time > end_time and current_time >= end_time:
                break
            elif next_ta_time > current_time:
                current_time = next_ta_time
            else:
                current_time += ta_interval
            
            iteration += 1
        
        simulation_results['total_iterations'] = (
            simulation_results['successful_iterations'] + simulation_results['failed_iterations']
        )
        
        logger.info("Main simulation loop completed")
        return simulation_results
        
    except Exception as e:
        logger.error(f"Simulation loop failed: {e}", exc_info=True)
        simulation_results['error'] = str(e)
        return simulation_results


def main():
    """
    Step 6.1: Main entry point for simulation.
    """
    try:
        # Generate execution ID
        execution_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        logger = setup_logging(execution_id)
        
        logger.info(f"Starting end-to-end simulation with execution ID: {execution_id}")
        
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration")
        config_path = str(project_root / "simulator" / "config" / "sim_config.yaml")
        config = load_config(config_path)
        
        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"Warehouse: {config['global']['wh_id']}")
        logger.info(f"Time range: {config['global']['start_time']} to {config['global']['end_time']}")
        
        # Step 2: Setup database and load external data
        logger.info("Step 2: Setting up database and loading external data")
        
        data_store_path = project_root / "simulator" / "data_store"
        
        # Step 2.1: Create database
        db_path = create_simulation_database(execution_id, data_store_path, logger)
        
        # Step 2.2: Load external data
        external_data_success = load_external_data(config_path, db_path, execution_id, logger)
        if not external_data_success:
            logger.error("External data loading failed - stopping simulation")
            return 1
        
        # Step 2.3: Validate database
        db_validation_success = validate_database(db_path, execution_id, logger)
        if not db_validation_success:
            logger.error("Database validation failed - stopping simulation")
            return 1
        
        logger.info("Database setup and external data loading completed successfully")
        
        # Step 3-5: Run simulation
        logger.info("Step 5: Running simulation")
        results = run_simulation(config, db_path, execution_id, logger)
        
        # Step 6: Display results summary
        logger.info("Step 6: Simulation completed")
        logger.info("=" * 60)
        logger.info("SIMULATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Execution ID: {results['execution_id']}")
        logger.info(f"Total iterations: {results['total_iterations']}")
        logger.info(f"Successful iterations: {results['successful_iterations']}")
        logger.info(f"Failed iterations: {results['failed_iterations']}")
        logger.info(f"TF iterations: {len(results['tf_results'])}")
        logger.info(f"TA iterations: {len(results['ta_results'])}")
        
        if 'error' in results:
            logger.error(f"Simulation error: {results['error']}")
            return 1
        
        if results['failed_iterations'] > 0:
            logger.warning("Simulation completed with some failures")
            return 1
        
        logger.info("Simulation completed successfully")
        return 0
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        if 'logger' in locals():
            logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main()) 