"""
Tour formation solver service module.

Provides service layer for tour formation solving.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from .tf_data import prepare_model_data, ModelData
from .tf_solver import TourFormationSolver, TourFormationResult

# Get module-specific logger
logger = logging.getLogger(__name__)

class TourFormationSolverService:
    """
    Manages the solving process for individual tour formation clusters.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, output_dir: str):
        """
        Initialize the solver service.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            logger (logging.Logger): Logger instance.
            output_dir (str): Directory to save output files.
        """
        self.config = config
        self.logger = logger
        self.output_dir = output_dir

        self.solver = TourFormationSolver(config=self.config, logger=self.logger, output_dir=self.output_dir)
        logger.debug("Initializing TourFormationSolverService")

    def solve_one_cluster(
        self,
        container_ids: List[str],
        containers_df: pd.DataFrame,
        skus_df: pd.DataFrame,
        planning_timestamp: datetime, 
        cluster_id: str,
        cluster_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data, build model, solve, and process results for a single cluster.

        Returns a dictionary representing the result, including status.
        """
        try:
            self.logger.info(f"[Solver Service] Solving cluster: {cluster_id}")

            # --- Save Cluster Metadata ---
            try:
                try:
                    metadata_df = pd.DataFrame([cluster_metadata]) 
                    write_index = False
                except ValueError:
                    # Fallback for non-scalar values
                    self.logger.debug(f"Metadata for cluster {cluster_id} not flat, saving as key-value pairs.")
                    metadata_df = pd.Series(cluster_metadata).to_frame(name='value')
                    metadata_df.index.name = 'key'
                    write_index = True

                metadata_path = os.path.join(self.output_dir, f"{cluster_id}_cluster_metadata.csv")
                metadata_df.to_csv(metadata_path, index=write_index)
                self.logger.debug(f"Saved cluster metadata to {metadata_path}")
            except Exception as meta_e:
                self.logger.error(f"Failed to save metadata for cluster {cluster_id}: {meta_e}")

            # --- Data Preparation ---
            # Extract num_tours from metadata
            num_tours = cluster_metadata.get('num_tours')
            if num_tours is None:
                self.logger.warning(f"'num_tours' not found in cluster metadata for {cluster_id}. Cannot prepare data.")
                # Handle error case - return a specific status
                return {
                    'cluster_id': cluster_id,
                    'status': 'Error_Missing_Metadata',
                    'tours': [],
                    'skipped_containers': container_ids,
                    'container_assignments': {},
                    'solve_time': 0,
                    'metadata': cluster_metadata
                }
                
            model_data: ModelData = prepare_model_data(
                # Pass the correct arguments based on tf_data.py definition
                container_data=containers_df, 
                slotbook_data=skus_df,     
                container_ids=container_ids, 
                num_tours=num_tours,       
                logger=self.logger
            )
            if not model_data or not model_data.container_ids: # Check if data prep yielded valid data
                self.logger.warning(f"[Solver Service] Skipping cluster {cluster_id} due to insufficient data after preparation.")
                # Return a standardized 'skipped' result
                return {
                    'cluster_id': cluster_id,
                    'status': 'Skipped',
                    'tours': [],
                    'skipped_containers': container_ids,
                    'container_assignments': {},
                    'solve_time': 0,
                    'metadata': cluster_metadata
                }

            # --- Solving ---
            # Get tour_id_offset, default to 0 if not present
            tour_id_offset = cluster_metadata.get('tour_id_offset', 0)
            self.logger.debug(f"Using tour_id_offset: {tour_id_offset} for cluster {cluster_id}")

            result: Optional[TourFormationResult] = self.solver.solve(
                model_data=model_data, 
                cluster_id=cluster_id,
                tour_id_offset=tour_id_offset 
            )

            # --- Result Processing ---
            if result and (result.status == 'Optimal' or result.status == 'Feasible') and result.aisle_ranges:
                self.logger.info(f"[Solver Service] Successfully solved cluster {cluster_id} with {len(result.aisle_ranges)} tours. Status: {result.status}")

                final_tours_data = result.aisle_ranges # Placeholder 
                skipped_containers = [] # Placeholder 
                
                return {
                    'cluster_id': cluster_id,
                    'status': result.status,
                    'tours': list(final_tours_data.items()), 
                    'skipped_containers': skipped_containers, # Populate correctly
                    'container_assignments': result.container_assignments,
                    'solve_time': result.solve_time,
                    'metadata': cluster_metadata
                }
            else:
                status = result.status if result else 'Failure'
                solve_time = result.solve_time if result else 0
                self.logger.warning(f"[Solver Service] Solver failed or found no solution for cluster {cluster_id}. Status: {status}")
                return {
                    'cluster_id': cluster_id,
                    'status': status,
                    'tours': [],
                    'skipped_containers': container_ids,
                    'container_assignments': {},
                    'solve_time': solve_time,
                    'metadata': cluster_metadata
                }

        except Exception as e:
            self.logger.error(f"[Solver Service] Error solving cluster {cluster_id}: {str(e)}", exc_info=True)
            return {
                'cluster_id': cluster_id,
                'status': 'Error',
                'tours': [],
                'skipped_containers': container_ids,
                'container_assignments': {},
                'solve_time': 0,
                'error_message': str(e),
                'metadata': cluster_metadata
            } 