"""
Optimization model for tour allocation.

This module handles the core optimization model including:
- Variable creation
- Constraint formulation
- Objective function definition
- Solution extraction
"""

from typing import Dict, Any, Optional
import gurobipy as gp
from gurobipy import GRB, Model
import logging

from .ta_data import ModelData
from pick_optimization.utils.logging_config import get_logger

# Get module-specific logger with workflow logging
logger = get_logger(__name__, 'tour_allocation')

class TourAllocationModel:
    """Optimization model for tour allocation problem."""
    
    def __init__(self, model_data: ModelData, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the optimization model.
        
        Parameters
        ----------
        model_data : ModelData
            Preprocessed data for the model
        config : Dict[str, Any]
            Configuration dictionary
        logger : logging.Logger
            Logger instance
        """
        self.model_data = model_data
        self.config = config
        self.logger = logger
        
        # Extract parameters from config
        alloc_config = config['tour_allocation']
        weights = alloc_config['weights']
        self.concurrency_weight = weights['concurrency']
        self.slack_weight = weights['slack']
        
        # Gurobi configs
        self.output_flag = alloc_config['solver']['output_flag']
        self.mip_gap = alloc_config['solver']['mip_gap']
        self.time_limit = alloc_config['solver']['time_limit']
        
        # Initialize Gurobi parameters
        gurobi_config = config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "GURO_PAR_ISVNAME": gurobi_config.get('ISV_NAME'),
            "GURO_PAR_ISVAPPNAME": gurobi_config.get('APP_NAME'),
            "GURO_PAR_ISVEXPIRATION": gurobi_config.get('EXPIRATION'),
            "GURO_PAR_ISVKEY": gurobi_config.get('CODE')
        }
        
        # Filter out None values to avoid encode errors
        self.gurobi_params = {k: v for k, v in self.gurobi_params.items() if v is not None}
        
        self.model = None
        self.solution = None
        
    def build(self) -> None:
        """Build the optimization model with all variables, constraints, and objective."""
        self.logger.info("Building optimization model...")
        
        try:
            # Create Gurobi environment and model
            self.solver_env = gp.Env(params=self.gurobi_params)
            self.model = Model("TourAllocation", env=self.solver_env)
            self.model.setParam("MIPGap", self.mip_gap)
            self.model.setParam("TimeLimit", self.time_limit)
            
            # Add variables
            self._add_variables()
            
            # Add constraints
            self._add_constraints()
            
            # Set objective
            self._set_objective()
            
            self.model.update()
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise
            
    def _add_variables(self) -> None:
        """Add all decision variables to the model."""
        m = self.model
        data = self.model_data
        
        # 1. Tour-Buffer assignment variables (y_kb)
        self.y = {}
        for k in data.tours:
            for b in range(data.max_buffer_spots):
                self.y[k,b] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f'y_{k}_{b}'
                )
        
        # 2. Aisle-Buffer/Picker assignment variables (z_ab)
        self.z = {}
        for a in data.aisles:
            for b in range(data.max_buffer_spots):
                self.z[a,b] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f'z_{a}_{b}'
                )
                
        # 3. Aisle concurrency variables (c_a)
        self.c = {}
        for a in data.aisles:
            self.c[a] = m.addVar(
                vtype=GRB.INTEGER,
                name=f'c_{a}'
            )
            
        m.update()
        self.logger.info(f"Created {m.NumVars} variables")
        
    def _add_constraints(self) -> None:
        """Add all constraints to the model."""
        self._add_unique_assignment_constraints()
        self._add_one_tour_per_buffer_constraints()
        self._add_aisle_visit_linking_constraints()
        self._add_concurrency_constraints()
        self._add_total_assignment_constraints()  
        
        self.model.update()
        self.logger.info(f"Added {self.model.NumConstrs} constraints")

    def _add_total_assignment_constraints(self) -> None:
        """Add constraint to ensure total assignments equals min(buffer_spots, tours)."""
        m = self.model
        data = self.model_data
        
        # Calculate minimum of buffer spots and number of tours
        min_assignments = min(data.max_buffer_spots, len(data.tours))
        
        # Add constraint that sum of all y[k,b] equals min_assignments
        m.addConstr(
            gp.quicksum(self.y[k,b] 
                        for k in data.tours 
                        for b in range(data.max_buffer_spots)) == min_assignments,
            name='total_assignments'
        )
        
    def _add_unique_assignment_constraints(self) -> None:
        """Add constraints ensuring each tour is assigned to maximum of one buffer spot."""
        m = self.model
        data = self.model_data
        
        for k in data.tours:
            m.addConstr(
                gp.quicksum(self.y[k,b] for b in range(data.max_buffer_spots)) <= 1,
                name=f'unique_assignment_{k}'
            )
            
    def _add_one_tour_per_buffer_constraints(self) -> None:
        """Add constraints ensuring each buffer spot gets at most one tour."""
        m = self.model
        data = self.model_data
        
        for b in range(data.max_buffer_spots):
            m.addConstr(
                gp.quicksum(self.y[k,b] for k in data.tours) <= 1,
                name=f'one_tour_per_buffer_{b}'
            )
            
    def _add_aisle_visit_linking_constraints(self) -> None:
        """Add constraints linking tour assignments to aisle visits."""
        m = self.model
        data = self.model_data
        
        for a in data.aisles:
            for b in range(data.max_buffer_spots):
                for k in data.tours:
                    if data.tour_aisle_visits[k,a] == 1:
                        m.addConstr(
                            self.z[a,b] >= self.y[k,b],
                            name=f'link_aisle_visit_{a}_{b}_{k}'
                        )
                        
    def _add_concurrency_constraints(self) -> None:
        """Add constraints defining aisle concurrency."""
        m = self.model
        data = self.model_data
        
        for a in data.aisles:
            # Get base concurrency for this aisle (default to 0 if not present)
            base_tour_count = data.aisle_concurrency.get(a, 0)
            
            # Concurrency definition - c[a] represents total concurrency
            m.addConstr(
                self.c[a] == base_tour_count + gp.quicksum(self.z[a,b] for b in range(data.max_buffer_spots)) - 1,
                name=f'concurrency_def_{a}'
            )
    
         
    def _set_objective(self) -> None:
        """Set the multi-component objective function."""
        m = self.model
        data = self.model_data
        
        # 1. Concurrency component
        self.concurrency = gp.quicksum(self.c[a] for a in data.aisles)
        
        # 2. Total slack component
        self.total_slack = gp.quicksum(
            data.total_slack[k] * self.y[k,b]
            for k in data.tours
            for b in range(data.max_buffer_spots)
        )
        
        # Set complete objective
        m.setObjective(
            - self.slack_weight * self.total_slack + self.concurrency_weight * self.concurrency,
            GRB.MINIMIZE
        )
        
        m.update()
        
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Solve the optimization model and return solution.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing solution components if optimal solution found,
            None otherwise
        """
        if self.model is None:
            raise ValueError("Model must be built before solving")
            
        try:
            self.logger.info("Solving optimization model...")
            self.model.optimize()
            
            if self.model.status == GRB.OPTIMAL:
                self.logger.info(f"Optimal solution found with objective: {self.model.objVal}")
                self.solution = self._extract_solution()
                self._log_solution_metrics()
                return self.solution
            elif self.model.status == GRB.INFEASIBLE:
                self._compute_and_log_iis()
                return None
            else:
                self.logger.warning(f"No optimal solution found. Status: {self.model.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error solving model: {str(e)}")
            raise
            
    def _extract_solution(self) -> Dict[str, Any]:
        """Extract solution components from the optimized model."""
        data = self.model_data
        
        solution = {
            'tour_assignments': {},
            'aisle_assignments': {},
            'concurrency': {},
            'objective_value': self.model.objVal,
            'metrics': {}
        }
        
        # Extract tour-buffer assignments
        for k in data.tours:
            for b in range(data.max_buffer_spots):
                if self.y[k,b].X > 0.5:
                    solution['tour_assignments'][k] = b
        
        # Extract aisle assignments and concurrency
        for a in data.aisles:
            solution['aisle_assignments'][a] = [
                b for b in range(data.max_buffer_spots)
                if self.z[a,b].X > 0.5
            ]
            solution['concurrency'][a] = int(self.c[a].X)
        
        # Calculate metrics
        solution['metrics'] = {
            'total_concurrency': sum(solution['concurrency'].values()),
            'total_slack': self.total_slack.getValue(),
            'max_concurrency': max(solution['concurrency'].values()),
            'empty_buffer_spots': data.max_buffer_spots - len(set(solution['tour_assignments'].values())),
            'total_aisle_visits': sum(len(buffers) for buffers in solution['aisle_assignments'].values())
        }
        
        return solution
        
    def _log_solution_metrics(self) -> None:
        """Log detailed metrics about the solution."""
        if not self.solution:
            return
            
        # Calculate component values
        concurrency_value = self.concurrency.getValue()
        total_slack_value = self.total_slack.getValue()
        
        self.logger.info("Tour Allocation Results:")
        self.logger.info(f"Total Objective Value: {self.solution['objective_value']:.2f}")
        self.logger.info("\nComponent Values:")
        self.logger.info(f"  - Concurrency (alpha={self.concurrency_weight}): {concurrency_value:.2f}")
        self.logger.info(f"  - Slack (beta={self.slack_weight}): {total_slack_value:.2f} hours")
    
        self.logger.info(f"  - Total Tours Assigned: {len(self.solution['tour_assignments'])}")
        self.logger.info(f"  - Empty Buffer Spots: {self.solution['metrics']['empty_buffer_spots']} out of {self.model_data.max_buffer_spots}")
        self.logger.info(f"  - Maximum Aisle Concurrency: {self.solution['metrics']['max_concurrency']}")
        
    def _compute_and_log_iis(self) -> None:
        """Compute and log the Irreducible Inconsistent Subsystem (IIS)."""
        self.logger.info("Computing IIS to identify problematic constraints...")
        self.model.computeIIS()
        
        # Initialize counters for different constraint types
        constraint_groups = {}
        bound_violations = {
            'lower_bounds': 0,
            'upper_bounds': 0
        }
        
        # Group constraints by their prefix
        for c in self.model.getConstrs():
            if c.IISConstr:
                constraint_type = c.ConstrName.split('_')[0]
                constraint_groups[constraint_type] = constraint_groups.get(constraint_type, 0) + 1
                
                self.logger.debug(f'\t{c.ConstrName}: {self.model.getRow(c)} {c.Sense} {c.RHS}')
        
        # Count bound violations
        for v in self.model.getVars():
            if v.IISLB:
                bound_violations['lower_bounds'] += 1
            if v.IISUB:
                bound_violations['upper_bounds'] += 1
        
        # Log summary of infeasible constraint groups
        self.logger.info("\nInfeasibility Summary by Constraint Type:")
        
        if constraint_groups:
            self.logger.info("\nConstraint Groups Contributing to Infeasibility:")
            for group, count in sorted(constraint_groups.items()):
                self.logger.info(f"\t{group}: {count} constraints")
        
        if any(bound_violations.values()):
            self.logger.info("\nBound Violations:")
            for bound_type, count in bound_violations.items():
                if count > 0:
                    self.logger.info(f"\t{bound_type.replace('_', ' ').title()}: {count} violations")