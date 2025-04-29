"""
Main solver module for tour formation.

This module provides the interface for solving individual tour formation problems,
handling the optimization model and solution generation. The solver supports
reading input from CSV files and writing solutions to disk for containerized execution.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd

# Local imports
from .tf_data import ModelData
from .tf_model import TourFormationModel

# Get module-specific logger
logger = logging.getLogger(__name__)

@dataclass
class TourFormationResult:
    """
    Container for tour formation results.
    
    Attributes
    ----------
    container_assignments : Dict[str, Dict[str, Any]]
        Mapping of container_id to assignment details including tour
    pick_assignments : Dict[str, List[Dict[str, Any]]]
        Mapping of container_id to list of pick details (sku, aisle, quantity)
    aisle_ranges : Dict[int, Dict[str, int]]
        Mapping of tour_id to aisle range details (min_aisle, max_aisle)
        Used as source data for the 'tours' output in API responses
    metrics : Dict[str, float]
        Optimization metrics including objective value components
    solve_time : float
        Time taken to solve the optimization problem in seconds
    cluster_id : str
        Identifier for the cluster that was solved
    status : str
        Solution status (e.g. "Optimal", "Infeasible", "Error")
    """
    container_assignments: Dict[str, Dict[str, Any]]
    pick_assignments: Dict[str, List[Dict[str, Any]]]
    aisle_ranges: Dict[int, Dict[str, int]]
    metrics: Dict[str, float]
    solve_time: float
    cluster_id: str
    status: str

class TourFormationSolver:
    """
    Main solver interface for tour formation problem.
    
    This class uses prepared ModelData and configuration to solve
    a single tour formation subproblem (cluster).
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: Optional[logging.Logger] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the solver.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        logger : Optional[logging.Logger], optional
            Logger instance, by default None
        output_dir : Optional[str], optional
            Directory to save output files, by default None
        """
        self.config = config
        self.output_dir = output_dir
        
       
        self.logger = logger
        
        solver_config = config.get('tour_formation', {}).get('solver', {})
        self.output_flag = solver_config.get('output_flag', 0)
        self.mip_gap = solver_config.get('mip_gap', 0.01)
        self.time_limit = solver_config.get('time_limit', 60)
        
        gurobi_config = config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "MIPGap": self.mip_gap,
            "TimeLimit": self.time_limit,
            **{k: v for k, v in {
                "GURO_PAR_ISVNAME": gurobi_config.get('ISV_NAME'),
                "GURO_PAR_ISVAPPNAME": gurobi_config.get('APP_NAME'),
                "GURO_PAR_ISVEXPIRATION": gurobi_config.get('EXPIRATION'),
                "GURO_PAR_ISVKEY": gurobi_config.get('CODE')
            }.items() if v is not None}
        }
        
        self.model_data = None
        self.model = None
        
    def solve(self, model_data: ModelData, cluster_id: str, tour_id_offset: int = 0) -> Optional[TourFormationResult]:
        """
        Solve a single tour formation problem using the provided model data.
        
        Args:
            model_data (ModelData): Prepared data for the optimization model.
            cluster_id (str): The identifier for the cluster being solved.
            tour_id_offset (int, optional): The offset for tour IDs. Defaults to 0.

        Returns:
        -------
        Optional[TourFormationResult]
            Solution result object if successful, None otherwise
        """
        solve_status = "Error" # Default status
        solve_time = 0.0
        try:
            start_time = time.time()
            self.logger.info(f"Solving subproblem for cluster: {cluster_id}") 
            
            # Build and solve model
            model_builder = TourFormationModel(
                config=self.config, 
                model_data=model_data, 
                gurobi_params=self.gurobi_params, 
                logger=self.logger,
                output_dir=self.output_dir, 
                cluster_id=cluster_id,      
                tour_id_offset=tour_id_offset 
            )
            model_builder.build_model()
            raw_solution = model_builder.solve() 
            solve_time = time.time() - start_time
            
            # Determine status after solving
            if raw_solution is not None and raw_solution: 
                solve_status = "Optimal" 
            elif raw_solution is None: 
                solve_status = "Infeasible or Error" 
            else: 
                solve_status = "No Active Tours"

            if raw_solution is not None:
                self.logger.info(f"Solution processing finished for cluster {cluster_id} with status {solve_status} in {solve_time:.2f}s.")
                result = self.create_result(raw_solution, solve_status, solve_time, cluster_id)
                self._log_solution_summary(result)
                return result
            else:
                self.logger.warning(f"No feasible solution found for cluster {cluster_id}. Status: {solve_status}. Time: {solve_time:.2f}s")
                return TourFormationResult(
                    container_assignments={},
                    pick_assignments={},
                    aisle_ranges={},
                    metrics={},
                    solve_time=solve_time,
                    cluster_id=cluster_id, 
                    status=solve_status 
                )
                
        except Exception as e:

            log_cluster_id = cluster_id if cluster_id else "unknown"
            self.logger.error(f"Error during tour formation solving for cluster {log_cluster_id}: {str(e)}", exc_info=True)
            # Return a result object indicating an error
            return TourFormationResult(
                container_assignments={},
                pick_assignments={},
                aisle_ranges={},
                metrics={},
                solve_time=solve_time, 
                cluster_id=log_cluster_id, 
                status="Error"
            )
        
    def create_result(
        self,
        solution: Dict[str, Any],
        status: str,
        solve_time: float,
        cluster_id: str 
    ) -> TourFormationResult:
        """Creates a TourFormationResult object from the raw solution dictionary."""
        return TourFormationResult(
            container_assignments=solution.get('container_assignments', {}),
            pick_assignments=solution.get('pick_assignments', {}),
            aisle_ranges=solution.get('aisle_ranges', {}),
            metrics=solution.get('metrics', {}),
            solve_time=solve_time,
            cluster_id=cluster_id, 
            status=status
        )
        
    def _log_solution_summary(self, result: TourFormationResult) -> None:
        """Logs a brief summary of the solution result."""
        self.logger.info("\nTour Formation Solution Summary:")
        self.logger.info(f"Cluster ID: {result.cluster_id}")
        self.logger.info(f"Containers assigned: {len(result.container_assignments)}")
        self.logger.info(f"Total picks: {sum(len(picks) for picks in result.pick_assignments.values())}")
        self.logger.info(f"Number of tours: {len(result.aisle_ranges)}")
        self.logger.info(f"Solve time: {result.solve_time:.2f} seconds")
        
        self.logger.info("\nMetrics:")
        for metric_name, metric_value in result.metrics.items():
            self.logger.info(f"{metric_name}: {metric_value:.2f}") 

def solve_tour_formation(
    containers_df: pd.DataFrame,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Solve the tour formation optimization problem."""
    try:
        logger = logger or logging.getLogger(__name__)
        logger.info("Starting tour formation solver")
        # ... existing code ...
    except Exception as e:
        logger.error(f"Error during tour formation solving: {str(e)}", exc_info=True)
        return {} 