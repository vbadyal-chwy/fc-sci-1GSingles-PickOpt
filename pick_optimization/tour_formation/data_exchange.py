"""
Data exchange module for tour formation.

This module handles reading and writing of subproblems and solutions
in standardized CSV format.
Most functions would need to be modified to support S3 input/output, but the core logic
would remain the same.
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

# Get module-specific logger with workflow logging
from logging_config import get_logger
logger = get_logger(__name__, 'tour_formation')

@dataclass
class SubproblemData:
    """Data class for holding tour formation subproblem data."""
    cluster_id: int
    container_ids: List[str]
    num_tours: int
    tour_id_offset: int
    creation_timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate subproblem data for completeness and consistency.
        
        Returns
        -------
        bool
            True if valid, raises ValueError otherwise
        """
        if not self.container_ids:
            raise ValueError("Container IDs list cannot be empty")
            
        if self.num_tours <= 0:
            raise ValueError(f"Invalid number of tours: {self.num_tours}")
            
        if self.tour_id_offset < 0:
            raise ValueError(f"Invalid tour ID offset: {self.tour_id_offset}")
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
def normalize_path(path: str) -> str:
    """
    Normalize path to use forward slashes.
    
    """
    return str(Path(path))

def load_container_data(
    input_dir: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Load container data from input directory."""
    try:
        logger = logger or logging.getLogger(__name__)
        logger.info("Loading container data")
        containers_path = os.path.join(input_dir, 'container_data.csv')
        
        if not os.path.exists(containers_path):
            msg = f"Containers file not found: {containers_path}"
            if logger:
                logger.error(msg)
            raise FileNotFoundError(msg)
        
        df = pd.read_csv(containers_path)
        
        # Convert datetime columns if they exist
        datetime_cols = ['arrive_datetime', 'pull_datetime', 'release_time']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        if logger:
            logger.debug(f"Loaded {len(df)} container records from {containers_path}")
        return df
        
    except Exception as e:
        msg = f"Error loading containers data: {str(e)}"
        if logger:
            logger.error(msg)
        raise

def load_cached_containers_with_slack(working_dir: str, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """
    Load cached container data with slack from the working directory.
    
    Parameters
    ----------
    working_dir : str
        Directory containing the cached containers file
    logger : Optional[logging.Logger]
        Logger instance for logging messages
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing container data with slack if successful, None otherwise
    """
    cached_containers_path = os.path.join(working_dir, 'containers_with_slack.csv')
    
    if not os.path.exists(cached_containers_path):
        if logger:
            logger.warning("Cached container data file not found. Will calculate slack.")
        return None
    
    try:
        if logger:
            logger.debug("Attempting to load cached container data with slack from the working directory")
        
        df = pd.read_csv(cached_containers_path)
        
        # Attempt to parse datetime columns if they were saved as strings
        for col in ['arrive_datetime', 'pull_datetime', 'release_time']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as dt_err:
                    if logger:
                        logger.warning(f"Could not parse datetime column '{col}' from cached file: {dt_err}")
        
        if logger:
            logger.info("Successfully loaded cached container data with slack.")
        return df
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load cached container data: {e}. Will calculate slack.")
        return None

def write_cached_containers_with_slack(
    working_dir: str,
    containers_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Write container data with slack to cache in the working directory.
    
    Parameters
    ----------
    working_dir : str
        Directory to write the cached containers file
    containers_df : pd.DataFrame
        DataFrame containing container data with slack
    logger : Optional[logging.Logger]
        Logger instance for logging messages
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        cached_containers_path = os.path.join(working_dir, 'containers_with_slack.csv')
        if logger:
            logger.debug(f"Saving container data with calculated slack to working directory: {cached_containers_path}")
        
        # Ensure working_dir exists before saving
        os.makedirs(working_dir, exist_ok=True)
        containers_df.to_csv(cached_containers_path, index=False)
        
        if logger:
            logger.info("Successfully saved container data with slack to working directory.")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to save container data with slack to working directory: {e}", exc_info=True)
        return False

def load_slotbook_data(input_dir: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load SKUs data from input directory.
    
    """
    skus_path = os.path.join(input_dir, 'slotbook_data.csv')
    
    if not os.path.exists(skus_path):
        msg = f"SKUs file not found: {skus_path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)
    
    try:
        df = pd.read_csv(skus_path)
        
        if logger:
            logger.debug(f"Loaded {len(df)} SKU records from {skus_path}")
        return df
        
    except Exception as e:
        msg = f"Error loading SKUs data: {str(e)}"
        if logger:
            logger.error(msg)
        raise

def write_subproblems(
    working_dir: str,
    output_dir: str,
    clusters: Union[Dict[str, List[str]], SubproblemData],
    cluster_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Write subproblems to CSV files.
    
    Parameters
    ----------
    working_dir : str
        Directory to write subproblem files
    clusters : Union[Dict[str, List[str]], SubproblemData]
        Dictionary mapping cluster IDs to container IDs or a SubproblemData object
    cluster_metadata : Optional[Dict[str, Dict[str, Any]]]
        Dictionary mapping cluster IDs to metadata
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        
        # Handle the case where a single SubproblemData is provided
        if isinstance(clusters, SubproblemData):
            return _write_single_subproblem(working_dir, output_dir, clusters, logger)
        
        # Write subproblems index
        subproblems_df = pd.DataFrame([
            {
                'cluster_id': cluster_id,
                'num_containers': len(container_ids),
                **(cluster_metadata.get(cluster_id, {}) if cluster_metadata else {})
            }
            for cluster_id, container_ids in clusters.items()
        ])
        subproblems_df.to_csv(
            os.path.join(output_dir, 'clustering_metadata.csv'),
            index=False
        )
        
        # Write individual cluster files
        for cluster_id, container_ids in clusters.items():
            
            # Write container IDs
            pd.DataFrame({'container_id': container_ids}).to_csv(
                os.path.join(working_dir, f'cluster_{cluster_id}_containers.csv'),
                index=False
            )
            
            # Write metadata
            if cluster_metadata:
                metadata = cluster_metadata.get(cluster_id, {})
                pd.DataFrame([metadata]).to_csv(
                    os.path.join(working_dir, f'cluster_{cluster_id}_metadata.csv'),
                    index=False
                )
        
        if logger:
            logger.debug(f"Wrote {len(clusters)} subproblems to {working_dir}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing subproblems: {str(e)}")
        return False

def _write_single_subproblem(
    working_dir: str,
    output_dir: str,
    subproblem: SubproblemData,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Write a single subproblem to disk.
    
    Parameters
    ----------
    working_dir : str
        Directory to write subproblem files
    subproblem : SubproblemData
        Subproblem data object
    logger : Optional[logging.Logger]
        Logger instance
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Validate the subproblem data
        subproblem.validate()
        
        # Write container IDs
        pd.DataFrame({'container_id': subproblem.container_ids}).to_csv(
            os.path.join(output_dir, f'cluster_{subproblem.cluster_id}_containers.csv'),
            index=False
        )
        
        # Write metadata combining fields and additional metadata
        metadata_dict = {
            'cluster_id': subproblem.cluster_id,
            'num_tours': subproblem.num_tours,
            'tour_id_offset': subproblem.tour_id_offset,
            'creation_timestamp': subproblem.creation_timestamp,
            'num_containers': len(subproblem.container_ids)
        }
        
        # Add any additional metadata
        if subproblem.metadata:
            metadata_dict.update(subproblem.metadata)
            
        pd.DataFrame([metadata_dict]).to_csv(
            os.path.join(working_dir, f'cluster_{subproblem.cluster_id}_metadata.csv'),
            index=False
        )
        
        # Check if we need to update the main index file
        index_path = os.path.join(working_dir, 'clustering_metadata.csv')
        if os.path.exists(index_path):
            # Load existing index
            index_df = pd.read_csv(index_path)
            
            # Check if this cluster ID already exists
            existing = index_df[index_df['cluster_id'] == subproblem.cluster_id]
            
            if len(existing) > 0:
                # Update existing entry
                for key, value in metadata_dict.items():
                    index_df.loc[index_df['cluster_id'] == subproblem.cluster_id, key] = value
            else:
                # Add new entry
                index_df = pd.concat([index_df, pd.DataFrame([metadata_dict])], ignore_index=True)
                
            # Write updated index
            index_df.to_csv(index_path, index=False)
        else:
            # Create new index file
            pd.DataFrame([metadata_dict]).to_csv(index_path, index=False)
        
        if logger:
            logger.debug(f"Wrote subproblem {subproblem.cluster_id} to {working_dir}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error writing single subproblem: {str(e)}")
        return False

def read_subproblem(
    working_dir: str,
    cluster_id: int,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Reads subproblem data for a specific cluster ID from CSV files.
    """
    log = logger
    working_dir = normalize_path(working_dir)

    metadata_path = os.path.join(working_dir, f"cluster_{cluster_id}_metadata.csv")
    containers_path = os.path.join(working_dir, f"cluster_{cluster_id}_containers.csv")

    metadata = {}

    # Read Metadata from CSV
    if not os.path.exists(metadata_path):
        msg = f"Metadata file not found: {metadata_path}"
        log.error(msg)
        raise FileNotFoundError(msg)
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        if not metadata_df.empty:
             metadata = metadata_df.iloc[0].to_dict() 
        else:
             log.warning(f"Metadata file is empty: {metadata_path}")
             metadata = {}
             
    except Exception as e:
        msg = f"Error loading metadata from {metadata_path}: {str(e)}"
        log.error(msg, exc_info=True)
        raise

    # Read Containers from CSV
    if not os.path.exists(containers_path):
        msg = f"Container file not found: {containers_path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    try:
        containers_df = pd.read_csv(containers_path)
        container_ids = [str(container_id) for container_id in containers_df['container_id'].tolist()]
    except Exception as e:
        msg = f"Error loading containers data from {containers_path}: {str(e)}"
        log.error(msg, exc_info=True)
        raise

    subproblem = {
        'cluster_id': cluster_id,
        'container_ids': container_ids,
        'metadata': metadata
    }

    if logger:
        logger.debug(f"Read subproblem for cluster {cluster_id} from {working_dir}")
        
    return subproblem


def write_results(
    output_dir: str,
    results: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Write a list of cluster results to the output directory.

    For each result in the list, writes separate CSV files for its dataframes
    (container_assignments, pick_assignments, etc.) using the naming convention:
    `cluster_{cluster_id}_{dataframe_type}.csv`.

    """
    log = logger or logging.getLogger(__name__)
    all_successful = True # Track if all writes succeed

    if not results:
        log.warning("No results provided to write_results.")
        return False # Nothing to write

    processed_clusters = 0
    for result in results:
        cluster_id = result.get('cluster_id')
        status = result.get('status', 'unknown')

        if not cluster_id:
            log.warning("Skipping result entry due to missing 'cluster_id'.")
            all_successful = False # Consider this a partial failure
            continue

        # Skip writing files for errored/failed clusters, but log it
        if status.lower() in ['error', 'failure']:
            log.warning(f"Skipping file writing for failed/errored cluster: {cluster_id} (status: {status})")
            continue 

        log.debug(f"Writing result files for cluster: {cluster_id} (status: {status})")
        write_success = True 
        files_written_for_cluster = []

        # Extract dataframes
        ca_df = result.get('container_assignments_df')
        pa_df = result.get('pick_assignments_df')
        ar_df = result.get('aisle_ranges_df')
        m_df = result.get('metrics_df')
        ct_df = result.get('container_tours_df') # Optional

        # Define file mappings
        file_map = {
            'container_assignments': ca_df,
            'pick_assignments': pa_df,
            'aisle_ranges': ar_df,
            'metrics': m_df,
            'container_tours': ct_df
        }

        # Write each dataframe to its own file
        for file_suffix, df in file_map.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"cluster_{cluster_id}_{file_suffix}.csv"
                filepath = os.path.join(output_dir, filename)
                try:
                    df.to_csv(filepath, index=False)
                    files_written_for_cluster.append(filename)
                except Exception as e:
                    log.error(f"Failed to write {filepath} for cluster {cluster_id}: {e}", exc_info=True)
                    write_success = False
            # else: log if df is missing/empty? Optional.

        if write_success and files_written_for_cluster:
            log.debug(f"Successfully wrote {len(files_written_for_cluster)} files for cluster {cluster_id}")
            processed_clusters += 1
        elif not files_written_for_cluster:
             log.debug(f"No dataframes found or written for {cluster_id} (Status: {status}).")
             # Decide if this counts as a failure; currently it doesn't set all_successful = False
        else: # write_success is False
             all_successful = False # Mark overall process as partially failed

    log.debug(f"Finished writing results. Processed {processed_clusters} clusters.")
    return all_successful