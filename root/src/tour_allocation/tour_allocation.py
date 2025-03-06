import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import gurobipy as gp
from gurobipy import GRB, Model
from dataclasses import dataclass
from tabulate import tabulate
from utility.logging_config import setup_logging

@dataclass
class TourData:
    """Container for preprocessed tour data"""
    tours: List[int]                          # Active tour IDs
    aisles: List[int]                         # Unique aisles
    tour_aisle_visits: Dict[Tuple[int, int], int]  # Tour-aisle visit matrix
    tour_lateness: Dict[int, float]           # Lateness by tour

class TourAllocationSolver:
    """
    Optimization model for tour allocation problem.
    
    Handles the assignment of tours to pickers while minimizing
    aisle concurrency and considering lateness objectives.
    """
    
    def __init__(
        self,
        tours_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any]
    ):
        """
        Initialize the tour allocation solver.
        
        Parameters
        ----------
        tours_data : Dict[str, pd.DataFrame]
            Dictionary containing:
            - tour_metrics: Tour-level metrics
            - pick_assignments: Individual pick assignments
            - container_assignments: Container-tour assignments
        num_pickers : int
            Number of available pickers
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.tours_data = tours_data
        self.config = config
        
        # Extract parameters from config
        tour_config = config['tour_allocation']
        weights = tour_config['weights']
        self.concurrency_weight = weights['concurrency']
        self.lateness_weight = weights['lateness']
        
        #Gurobi configs
        self.output_flag = tour_config['solver']['output_flag']
        self.mip_gap = tour_config['solver']['mip_gap']
        self.time_limit = tour_config['solver']['time_limit']
        
        # Initialize Gurobi parameters
        gurobi_config = config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "GURO_PAR_ISVNAME": gurobi_config.get('ISV_NAME'),
            "GURO_PAR_ISVAPPNAME": gurobi_config.get('APP_NAME'),
            "GURO_PAR_ISVEXPIRATION": gurobi_config.get('EXPIRATION'),
            "GURO_PAR_ISVKEY": gurobi_config.get('CODE')
        }
        
        self.logger = setup_logging(config, 'tour_allocation')
        self.tour_data = None
        self.model = None
        self.solution = None
        
    def prepare_data(self) -> None:
        """
        Prepare data structures for optimization model.
        Process input data into required format.
        """
        self.logger.info("Preparing tour allocation data...")
        
        try:
            # Get active tours from tour metrics
            active_tours = []
            for _, tour in self.tours_data['tour_metrics'].iterrows():
                if tour['ContainerCount'] > 0:
                    active_tours.append(tour['TourID'])
            active_tours = sorted(active_tours)
            
            # Get aisles from pick assignments
            picks_df = self.tours_data['pick_assignments']
            aisles = picks_df[picks_df['TourID'].isin(active_tours)]['Aisle'].unique()
            aisles = sorted(list(aisles))
            
            # Create tour-aisle visit matrix
            tour_aisle_visits = {}
            for tour_id in active_tours:
                for aisle in aisles:
                    tour_aisle_visits[tour_id, aisle] = 0
                    
            for _, pick in picks_df[picks_df['TourID'].isin(active_tours)].iterrows():
                tour_aisle_visits[pick['TourID'], pick['Aisle']] = 1
            
            # Calculate lateness by tour
            tour_lateness = self.tours_data['container_assignments'].groupby(
                'TourID'
            )['Lateness'].sum().to_dict()
            
            self.tour_data = TourData(
                tours=active_tours,
                aisles=aisles,
                tour_aisle_visits=tour_aisle_visits,
                tour_lateness=tour_lateness
            )
            
            self.logger.info(
                f"Processed {len(active_tours)} active tours and {len(aisles)} aisles"
            )
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def build_model(self) -> None:
        """
        Build the optimization model with all variables,
        constraints, and objective function.
        """
        self.prepare_data()
        if self.tour_data is None:
            raise ValueError("Data must be prepared before building model")
            
        self.logger.info("Building tour allocation model...")
        
        try:
            # Create Gurobi environment and model
            self.solver_env = gp.Env(params=self.gurobi_params)
            self.model = Model("TourAllocation", env=self.solver_env)
            self.model.setParam("MIPGap",self.mip_gap)
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
            
    def solve(self,  tours_to_release: int) -> Optional[Dict[str, Any]]:
        """
        Solve the optimization model and return solution.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing solution components if optimal solution found,
            None otherwise
        """
        self.num_pickers = tours_to_release
        self.build_model()
        
        if self.model is None:
            raise ValueError("Model must be built before solving")
            
        try:
            self.logger.info("Solving tour allocation model...")
            self.model.optimize()
            
            if self.model.status == GRB.OPTIMAL:
                self.logger.info(f"Optimal solution found with objective: {self.model.objVal}")
                self.solution = self._extract_solution()
                self.log_solution_metrics(self.solution)
                return self.solution
            elif self.model.status == GRB.INFEASIBLE:
                self._compute_and_log_iis(self.model)
                return None
            else:
                self.logger.warning(f"No optimal solution found. Status: {self.model.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error solving model: {str(e)}")
            raise
            
    def _add_variables(self) -> None:
        """Add all decision variables to the model"""
        m = self.model
        data = self.tour_data
        
        # 1. Tour-Picker assignment variables (y_kr)
        self.y = {}
        for k in data.tours:
            for r in range(self.num_pickers):
                self.y[k,r] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f'y_{k}_{r}'
                )
        
        # 2. Aisle-Picker assignment variables (z_ar)
        self.z = {}
        for a in data.aisles:
            for r in range(self.num_pickers):
                self.z[a,r] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f'z_{a}_{r}'
                )
                
        # 3. Aisle concurrency variables (c_a)
        self.c = {}
        for a in data.aisles:
            self.c[a] = m.addVar(
                vtype=GRB.INTEGER,
                name=f'c_{a}'
            )
            
        m.update()
        self.logger.info(f"Created {m.NumVars} variables:")
        self.logger.debug(f"Tour-Picker assignment (y): {len(self.y)}")
        self.logger.debug(f"Aisle-Picker assignment (z): {len(self.z)}")
        self.logger.debug(f"Aisle concurrency (c): {len(self.c)}")
            
    def _add_constraints(self) -> None:
        """Add all constraints to the model"""
        self._add_unique_assignment_constraints()
        self._add_one_tour_per_picker_constraints()
        self._add_aisle_visit_linking_constraints()
        self._add_concurrency_constraints()
        
        self.model.update()
        self.logger.info(f"Added {self.model.NumConstrs} constraints")
    
    #C1 - Unique Assignment for Each Tour   
    def _add_unique_assignment_constraints(self) -> None:
        """Add constraints ensuring each tour is assigned to exactly one picker"""
        m = self.model
        data = self.tour_data
        
        for k in data.tours:
            m.addConstr(
                gp.quicksum(self.y[k,r] for r in range(self.num_pickers)) <= 1,
                name=f'unique_assignment_{k}'
            )
    
    #C2 - One Tour per Picker       
    def _add_one_tour_per_picker_constraints(self) -> None:
        """Add constraints ensuring each picker gets at most one tour"""
        m = self.model
        data = self.tour_data
        
        for r in range(self.num_pickers):
            m.addConstr(
                gp.quicksum(self.y[k,r] for k in data.tours) == 1,
                name=f'one_tour_per_picker_{r}'
            )
    
    #C3 - Aisle-Picker Linking      
    def _add_aisle_visit_linking_constraints(self) -> None:
        """Add constraints linking tour assignments to aisle visits"""
        m = self.model
        data = self.tour_data
        
        for a in data.aisles:
            for r in range(self.num_pickers):
                for k in data.tours:
                    if data.tour_aisle_visits[k,a] == 1:
                        m.addConstr(
                            self.z[a,r] >= self.y[k,r],
                            name=f'link_aisle_visit_{a}_{r}_{k}'
                        )
    
    #C4 - Congestion Measure                   
    def _add_concurrency_constraints(self) -> None:
        """Add constraints defining aisle concurrency"""
        m = self.model
        data = self.tour_data
        
        for a in data.aisles:
            # Concurrency definition
            m.addConstr(
                self.c[a] >= gp.quicksum(self.z[a,r] for r in range(self.num_pickers)) - 1,
                name=f'concurrency_def_{a}'
            )
            # Non-negativity
            m.addConstr(
                self.c[a] >= 0,
                name=f'concurrency_nonneg_{a}'
            )
            
    def _set_objective(self) -> None:
        """Set the multi-component objective function"""
        m = self.model
        data = self.tour_data
        
        # 1. Concurrency component
        self.concurrency = gp.quicksum(self.c[a] for a in data.aisles)
        
        # 2. Lateness component
        self.lateness = gp.quicksum(
            data.tour_lateness[k] * self.y[k,r]
            for k in data.tours
            for r in range(self.num_pickers)
        )
        
        # Set complete objective
        m.setObjective(
            self.concurrency_weight * self.concurrency +
            self.lateness_weight * self.lateness,
            GRB.MINIMIZE
        )
        
        m.update()
        
    def _extract_solution(self) -> Dict[str, Any]:
        """Extract solution details from optimized model"""
        data = self.tour_data
        
        solution = {
            'tour_assignments': {},
            'aisle_assignments': {},
            'concurrency': {},
            'objective_value': self.model.objVal,
            'metrics': {}
        }
        
        # Extract tour assignments
        for k in data.tours:
            for r in range(self.num_pickers):
                if self.y[k,r].X > 0.5:
                    solution['tour_assignments'][k] = r
        
        # Extract aisle assignments and concurrency
        for a in data.aisles:
            solution['aisle_assignments'][a] = [
                r for r in range(self.num_pickers)
                if self.z[a,r].X > 0.5
            ]
            solution['concurrency'][a] = int(self.c[a].X)
        
        # Calculate metrics
        solution['metrics'] = {
            'total_concurrency': sum(solution['concurrency'].values()),
            'max_concurrency': max(solution['concurrency'].values()),
            'active_pickers': len(set(solution['tour_assignments'].values())),
            'total_aisle_visits': sum(len(pickers) for pickers in solution['aisle_assignments'].values())
        }
        
        return solution
        
    # amazonq-ignore-next-line
    def generate_summary(self, solution: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Generate summary dataframes from solution.
        
        Parameters
        ----------
        solution : Dict[str, Any]
            Solution dictionary from solve method
                
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing summary dataframes
        """
        if solution is None:
            return {}
                
        # Create picker assignments dataframe
        picker_assignments = []
        for tour_id, picker_id in solution['tour_assignments'].items():
            tour_aisles = [a for a in self.tour_data.aisles 
                        if self.tour_data.tour_aisle_visits[tour_id, a] == 1]
            picker_assignments.append({
                'TourID': tour_id,
                'PickerID': picker_id,
                'NumAisles': len(tour_aisles),
                'MinAisle': min(tour_aisles) if tour_aisles else None,
                'MaxAisle': max(tour_aisles) if tour_aisles else None,
                'Lateness': self.tour_data.tour_lateness.get(tour_id, 0)
            })
        picker_assignments_df = pd.DataFrame(picker_assignments)
        
        # Create aisle metrics dataframe
        aisle_metrics = []
        for aisle in self.tour_data.aisles:
            assigned_pickers = solution['aisle_assignments'].get(aisle, [])
            if len(assigned_pickers) > 0:  # Only include aisles with assignments
                aisle_metrics.append({
                    'Aisle': aisle,
                    'NumPickers': len(assigned_pickers),
                    'AssignedPickers': ','.join(map(str, assigned_pickers)),
                    'Concurrency': solution['concurrency'].get(aisle, 0)
                })
        aisle_metrics_df = pd.DataFrame(aisle_metrics)
        
        # Create picker metrics dataframe
        picker_metrics = []
        for picker_id in range(self.num_pickers):
            assigned_tours = [
                tour_id for tour_id, assigned_picker 
                in solution['tour_assignments'].items()
                if assigned_picker == picker_id
            ]
            
            if assigned_tours:
                assigned_aisles = set()
                for tour_id in assigned_tours:
                    tour_aisles = [
                        a for a in self.tour_data.aisles 
                        if self.tour_data.tour_aisle_visits[tour_id, a] == 1
                    ]
                    assigned_aisles.update(tour_aisles)
                
                picker_metrics.append({
                    'PickerID': picker_id,
                    'NumTours': len(assigned_tours),
                    'AssignedTours': ','.join(map(str, assigned_tours)),
                    'UniqueAisles': len(assigned_aisles),
                    'AisleRange': f"{min(assigned_aisles)}-{max(assigned_aisles)}" 
                        if assigned_aisles else "None"
                })
                    
        picker_metrics_df = pd.DataFrame(picker_metrics)
        
        # Generate summary tables
        # 1. Tour Allocation Summary
        tour_summary = []
        # Sort by tour cut time
        picker_assignments_df = picker_assignments_df.sort_values(
            by=['TourID']
        )
        for _, row in picker_assignments_df.iterrows():
            tour_summary.append([
                f"Tour {row['TourID']}",
                f"Picker {row['PickerID']}",
                row['NumAisles'],
                f"{row['MinAisle']}-{row['MaxAisle']}",
                f"{row['Lateness']:.1f}"
            ])
        
        self.logger.info("\nTour Allocation Summary:")
        self.logger.info("\n" + tabulate(
            tour_summary,
            headers=['Tour', 'Assigned Picker', 'Aisle Count', 'Aisle Range', 'Lateness (hrs)'],
            tablefmt='grid'
        ))
        
        # 2. Aisle Concurrency Summary
        aisle_summary = []
        # Sort by aisle number
        aisle_metrics_df = aisle_metrics_df.sort_values(by=['Aisle'])
        for _, row in aisle_metrics_df.iterrows():
            if row['NumPickers'] > 0:  # Only show aisles with assigned pickers
                aisle_summary.append([
                    f"Aisle {row['Aisle']}",
                    row['NumPickers'],
                    row['AssignedPickers'],
                    row['Concurrency']
                ])
        
        if aisle_summary:  # Only show table if there are entries
            self.logger.info("\nAisle Concurrency Summary:")
            self.logger.info("\n" + tabulate(
                aisle_summary,
                headers=['Aisle', 'Picker Count', 'Assigned Pickers', 'Concurrency'],
                tablefmt='grid'
            ))
            
        return {
            'picker_assignments': picker_assignments_df,
            'aisle_metrics': aisle_metrics_df,
            'picker_metrics': picker_metrics_df,
            'summary_metrics': solution['metrics']
        }
        
    def log_solution_metrics(self, solution: Dict[str, Any]) -> None:
        """
        Log detailed metrics about the solution.
        
        Parameters
        ----------
        solution : Dict[str, Any]
            Solution dictionary from solve method
        """
        # Calculate component values
        concurrency_value = self.concurrency.getValue()
        lateness_value = self.lateness.getValue()
        
        self.logger.info("Tour Allocation Optimization Results:")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Objective Value: {solution['objective_value']:.2f}")
        self.logger.info("Component Values:")
        self.logger.info(f"  - Concurrency (alpha={self.concurrency_weight}): {concurrency_value:.2f}")
        self.logger.info(f"  - Lateness (beta={self.lateness_weight}): {lateness_value:.2f} hours")
        self.logger.info("Weighted Components:")
        self.logger.info(f"  - Weighted Concurrency: {(self.concurrency_weight * concurrency_value):.2f}")
        self.logger.info(f"  - Weighted Lateness: {(self.lateness_weight * lateness_value):.2f}")
        self.logger.info("Allocation Metrics:")
        self.logger.info(f"  - Total Tours Assigned: {len(solution['tour_assignments'])}")
        self.logger.info(f"  - Active Pickers Used: {solution['metrics']['active_pickers']} out of {self.num_pickers}")
        self.logger.info(f"  - Maximum Aisle Concurrency: {solution['metrics']['max_concurrency']}")

    def _compute_and_log_iis(self, model: gp.Model) -> None:
        """
        Compute and log the Irreducible Inconsistent Subsystem (IIS).
        Used for debugging when model is infeasible.
        
        Parameters
        ----------
        model : gp.Model
            The Gurobi model to analyze
        """
        self.logger.info("Computing IIS to identify problematic constraints...")
        model.computeIIS()
        
        # Initialize counters for different constraint types
        constraint_groups = {}
        bound_violations = {
            'lower_bounds': 0,
            'upper_bounds': 0
        }
        
        # Group constraints by their prefix
        for c in model.getConstrs():
            if c.IISConstr:
                constraint_type = c.ConstrName.split('_')[0]
                constraint_groups[constraint_type] = constraint_groups.get(constraint_type, 0) + 1
                
                self.logger.debug(f'\t{c.ConstrName}: {model.getRow(c)} {c.Sense} {c.RHS}')
        
        # Count bound violations
        for v in model.getVars():
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