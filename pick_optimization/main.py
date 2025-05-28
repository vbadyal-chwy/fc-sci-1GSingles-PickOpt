import sys
import json
import os
import boto3
import logging
import yaml
from pathlib import Path
from datetime import datetime
from tour_formation.tf_entry import run_tour_formation_entrypoint
from tour_allocation.ta_entry import run_tour_allocation_entrypoint
from logging_config import setup_logging

# Initialize basic logger - will be enhanced once config is loaded
logger = logging.getLogger(__name__)

def load_config(entry_type: str) -> dict:
    """Load configuration based on entry type."""
    config_file = f"pick_optimization/data/{entry_type}_config.yaml"
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using default logging config")
        return {
            'logging': {
                'log_base_dir': 'data/logs',
                'console_level': 'INFO',
                'file_level': 'DEBUG'
            }
        }
    except Exception as e:
        logger.error(f"Error loading config file {config_file}: {e}")
        raise

def main():
    # ------------This part is for local hosting the model only------------------
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py '<json_params>'")
    #     sys.exit(1)

    # try:
    #     params = json.loads(sys.argv[1])
    # except json.JSONDecodeError:
    #     print("Invalid JSON input")
    #     sys.exit(1)

    # entry_type = params.get("entry_point_type")
    # if entry_type == "tour_formation":
    #     run_tour_formation_entrypoint(params)
    # elif entry_type == "tour_allocation":
    #     run_tour_allocation_entrypoint(params)
    # else:
    #     print(f"Unknown entry_point_type: {entry_type}")
    #     sys.exit(1)
    
    # Get input parameters first to determine entry type for config loading
    s3_dir = Path("pick_optimization/s3_data")
    s3_dir.mkdir(parents=True, exist_ok=True)
    input_str = os.environ.get("ENTRY_PARAMS") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not input_str:
        # Use basic logger since we haven't initialized centralized logging yet
        print("Usage: pass JSON as env var ENTRY_PARAMS or as first argument.")
        sys.exit(1)

    try:
        params = json.loads(input_str)
    except json.JSONDecodeError:
        print("Invalid JSON input")
        sys.exit(1)

    entry_type = params.get("entry_point_type")
    
    # Load configuration and initialize centralized logging
    try:
        config = load_config(entry_type)
        # Initialize centralized logging with 'main' workflow
        global logger
        logger = setup_logging(config, 'main')
        logger.info("Centralized logging initialized for main workflow")
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        sys.exit(1)
    
    logger.info("Starting main function")
    logger.info(f"Entry type: {entry_type}")
    logger.info(f"Input parameters: {input_str}")

    simulation_id = params.get("simulation_id")

    download_s3(s3_dir, simulation_id)
    entry_point_params = params.get("params")
    wh_id = entry_point_params.get("fc_id")
    iso_string = entry_point_params.get("planning_timestamp")
    dt = datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S")
    formatted = dt.strftime("%Y%m%d_%H%M%S")
    if entry_type == "tour_formation":
        # from tour_formation.tf_entry import run_tour_formation_entrypoint
        logger.info(f"Starting tour formation with params: {params}")
        run_tour_formation_entrypoint(
            mode = entry_point_params.get("mode"),
            fc_id=wh_id,
            planning_timestamp=entry_point_params.get("planning_timestamp"),
            # input_dir=entry_point_params.get("input_dir"),
            input_dir=s3_dir / "input" / wh_id / formatted, 
            # output_dir=entry_point_params.get("output_dir"),
            output_dir=s3_dir / "output" / wh_id / formatted,
            # working_dir=entry_point_params.get("working_dir"),
            working_dir=s3_dir / "working" / wh_id,
            labor_headcount=entry_point_params.get("labor_headcount"),
            cluster_id=entry_point_params.get("cluster_id")
        )

    elif entry_type == "tour_allocation":
        # from tour_allocation.ta_entry import run_tour_allocation_entrypoint
        logger.info(f"Starting tour allocation with params: {params}")
        entry_point_params = params.get("params")
        run_tour_allocation_entrypoint(
            fc_id=entry_point_params.get("fc_id"),
            planning_timestamp=entry_point_params.get("planning_timestamp"),
            # input_dir=entry_point_params.get("input_dir"),
            input_dir=s3_dir / "input" / wh_id / formatted,
            # output_dir=entry_point_params.get("output_dir"),
            output_dir=s3_dir / "output"/ wh_id / formatted,
            target_tours=entry_point_params.get("target_tours")
        )
    else:
        logger.error(f"Unknown entry_point_type: {entry_type}")
        sys.exit(1)
            
    upload_s3(s3_dir, simulation_id)
    logger.info("Main function completed successfully")


def download_s3(local_s3_dir: str, simulation_id: str, bucket_name: str = "fc-pick-opt"):
    s3 = boto3.client('s3')

    s3_prefix = f"{simulation_id}/data/pick_optimization/"

    # List all objects with the prefix
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith((".csv", ".yaml")):
                continue  # Only download .csv files

            # Compute local file path
            relative_path = key[len(s3_prefix):]
            local_path = local_s3_dir / relative_path

            # Ensure the subdirectory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download
            try:

                s3.download_file(bucket_name, key, str(local_path))
                logger.info(f"Downloaded s3://{bucket_name}/{key} to {local_path}")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                continue

        
def upload_s3(local_s3_dir: str, simulation_id: str, bucket_name: str = "fc-pick-opt"):

    s3 = boto3.client('s3')

    # local path to sync
    if not local_s3_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_s3_dir}")

    # Base S3 prefix
    base_s3_prefix = f"{simulation_id}/data/pick_optimization/"

    for root, dirs, files in os.walk(local_s3_dir):
        for file in files:
            if file.endswith((".csv", ".yaml")):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(local_s3_dir)

                # Final S3 key
                s3_key = f"{base_s3_prefix}{relative_path.as_posix()}"

                # Upload
                s3.upload_file(str(file_path), bucket_name, s3_key)
                logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

if __name__ == "__main__":
    main()
