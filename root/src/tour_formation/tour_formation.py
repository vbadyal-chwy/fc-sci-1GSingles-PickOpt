import pandas as pd
from typing import Dict, List, Tuple, Any, Set
import gurobipy as gp
from gurobipy import GRB, Model
from datetime import datetime
from dataclasses import dataclass, field
from tabulate import tabulate
from utility.logging_config import setup_logging
import time
import math
# Import the existing functions from container_clustering.py
from tour_formation.container_clustering import modified_cluster_containers

@dataclass
class ModelData:
    """Container for preprocessed model data"""
    container_ids: List[str]
    skus: List[str]
    container_sku_qty: Dict[Tuple[str, str], int]
    sku_aisles: Dict[str, List[int]]
    sku_min_aisle: Dict[str, List[int]]
    sku_max_aisle: Dict[str, List[int]]
    aisle_inventory: Dict[Tuple[str, int], int]
    tour_indices: List[int]  
    max_aisle: int
    is_last_iteration: bool = True
    single_location_skus: Dict[str, int] = field(default_factory=dict)  # SKU -> unique aisle
    multi_location_skus: Dict[str, List[int]] = field(default_factory=dict)  # SKU -> list of aisles
    container_fixed_aisles: Dict[str, Dict[int, int]] = field(default_factory=dict)  # container -> {aisle -> qty}
    
class TourFormationSolver:
    """
    Optimization model for the tour formation problem.
    
    Handles the assignment of containers to tours and selecting
    pick locations while minimizing lateness and travel distance.
    """
    
    def __init__(self, container_data: pd.DataFrame, slotbook_data: pd.DataFrame, planning_timestamp: datetime,
        config: Dict[str, Any], num_tours: int
    ):
        """
        Initialize the tour formation solver.
        
        Parameters
        ----------
        container_data : pd.DataFrame
            Container data with order details
        slotbook_data : pd.DataFrame
            Slotbook data with inventory details
        planning_timestamp : datetime
            Current planning timestamp
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.container_data = container_data
        self.slotbook_data = slotbook_data
        self.planning_timestamp = planning_timestamp
        self.config = config
        
        # Extract parameters from config
        self.hourly_container_target = config['global']['hourly_container_target']
        pick_config = config['tour_formation']
        self.min_containers_per_tour = pick_config['min_containers_per_tour']
        self.max_containers_per_tour = pick_config['max_containers_per_tour']
        weights = pick_config['weights']
        self.alpha = weights['lateness']
        self.beta = weights['distinct_aisles']
        self.gamma = weights['tour_count']

        self.early_termination_seconds = pick_config['early_termination_seconds']
        self.max_cluster_size = pick_config['max_cluster_size']
        self.clustering_enabled = pick_config['clustering_enabled']
        self.num_tours = num_tours
        
        # Calculate number of tours based on container count
        #num_containers = len(container_data['container_id'].unique())
        #self.num_tours = math.ceil(num_containers /self.min_containers_per_tour)
        
        # Gurobi configs
        self.output_flag = pick_config['solver']['output_flag']
        self.mip_gap = pick_config['solver']['mip_gap']
        self.time_limit = pick_config['solver']['time_limit']

        # Initialize Gurobi parameters
        gurobi_config = self.config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "GURO_PAR_ISVNAME": gurobi_config['ISV_NAME'],
            "GURO_PAR_ISVAPPNAME": gurobi_config['APP_NAME'],
            "GURO_PAR_ISVEXPIRATION": gurobi_config['EXPIRATION'],
            "GURO_PAR_ISVKEY": gurobi_config['CODE']
        }
        
        self.logger = setup_logging(config, 'tour_formation')
        self.model_data = None
        self.model = None
        self.solution = None
        
        # Add new properties for sequential optimization
        self.processed_containers = set()
        self.tour_id_offset = 0
        
    def prepare_data(self, container_ids=None) -> None:
        """
        Prepare data structures for optimization model.
        Classify SKUs into single-location and multi-location groups.
        
        Parameters
        ----------
        container_ids : Optional[List[str]]
            Optional list of specific container IDs to include in the optimization
        """
        self.logger.info("Preparing data for optimization model...")
        
        try:
            # Filter container data if specific container IDs provided
            if container_ids is not None:
                filtered_container_data = self.container_data[
                    self.container_data['container_id'].isin(container_ids)
                ]
            else:
                filtered_container_data = self.container_data
                
            container_ids = filtered_container_data['container_id'].unique().tolist()
            skus = filtered_container_data['item_number'].unique().tolist()
            
            # Create container-SKU quantity mapping
            container_sku_qty = {}
            for _, row in filtered_container_data.iterrows():  # Use filtered data here
                container_sku_qty[(row['container_id'], row['item_number'])] = row['quantity']
            
            # Create SKU-Aisle mapping and classify SKUs by location count
            sku_aisles = {}
            max_aisle = self.slotbook_data['aisle_sequence'].max()
            
            # Classify SKUs by number of aisle locations
            single_location_skus = {}
            multi_location_skus = {}
            
            for sku in skus:
                sku_locs = self.slotbook_data[
                    self.slotbook_data['item_number'] == sku
                ]['aisle_sequence'].tolist()
                sorted_locs = sorted(sku_locs)
                sku_aisles[sku] = sorted_locs
                
                if len(sorted_locs) == 1:
                    single_location_skus[sku] = sorted_locs[0]
                else:
                    multi_location_skus[sku] = sorted_locs
            
            # Create mapping of fixed aisle requirements for each container
            container_fixed_aisles = {}
            for i in container_ids:
                fixed_aisles = {}
                for s in skus:
                    if (i, s) in container_sku_qty and s in single_location_skus:
                        a = single_location_skus[s]
                        fixed_aisles[a] = fixed_aisles.get(a, 0) + container_sku_qty[(i, s)]
                container_fixed_aisles[i] = fixed_aisles
            
            # Log optimization statistics
            single_count = len(single_location_skus)
            multi_count = len(multi_location_skus)
            total_skus = single_count + multi_count
            single_pct = (single_count / total_skus) * 100 if total_skus > 0 else 0
            
            self.logger.info(f"SKU Optimization: {single_count} SKUs ({single_pct:.1f}%) have single locations")
            self.logger.info(f"SKU Optimization: {multi_count} SKUs ({100-single_pct:.1f}%) have multiple locations")
            
            # Compute SKU-specific aisle bounds
            sku_min_aisle = {}
            sku_max_aisle = {}
            for s, aisles in sku_aisles.items():
                if aisles:  # Check if the list is not empty
                    sku_min_aisle[s] = min(aisles)
                    sku_max_aisle[s] = max(aisles)
        
            # Create aisle inventory mapping
            aisle_inventory = {}
            for _, row in self.slotbook_data.iterrows():
                aisle_inventory[(row['item_number'], row['aisle_sequence'])] = row['actual_qty']
            
            #self.num_tours = math.floor(len(container_ids) / self.max_containers_per_tour)
            
            # Define tour indices - adjust the number of tours based on filtered containers if needed
            if container_ids is not None:
                #adjusted_num_tours = max(1, math.floor(len(container_ids) / self.max_containers_per_tour))
                #adjusted_num_tours = max(0, int(math.floor(len(container_ids) / self.max_containers_per_tour)))      #bookmark
                tour_indices = list(range( self.num_tours))
            else:
                tour_indices = list(range(self.num_tours))
            
            
            self.logger.info(f"Generated a maximum of {len(tour_indices)} tours for {len(container_ids)} containers")
            
            
            # Store preprocessed data
            self.model_data = ModelData(
                container_ids=container_ids,
                skus=skus,
                container_sku_qty=container_sku_qty,
                sku_aisles=sku_aisles,
                sku_min_aisle=sku_min_aisle,
                sku_max_aisle=sku_max_aisle,
                aisle_inventory=aisle_inventory,
                tour_indices=tour_indices,
                max_aisle=max_aisle,
                single_location_skus=single_location_skus,
                multi_location_skus=multi_location_skus,
                container_fixed_aisles=container_fixed_aisles
            )
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
        
    def _get_slack_information(self) -> Tuple[bool, Set[str], Dict[str, float]]:
        """
        Extract slack information from container data if available.
        
        Returns
        -------
        Tuple[bool, Set[str], Dict[str, float]]
            - Boolean indicating if slack data is available
            - Set of container IDs
            - Dictionary of priority weights by container ID
        """
        # Check if slack data is available
        has_slack_data = 'slack_category' in self.container_data.columns
        critical_containers = set()
        priority_weights = {}
        
        if has_slack_data:
            # Get unique container-category pairs
            container_categories = self.container_data[['container_id', 'slack_category']].drop_duplicates()
            
            # Identify critical containers
            critical_df = container_categories[container_categories['slack_category'] == 'Critical']
            critical_containers = set(critical_df['container_id'])
            
            # Assign priority weights based on slack category
            for _, row in container_categories.iterrows():
                container_id = row['container_id']
                category = row['slack_category']
                
                # Assign weights based on category
                if category == 'Critical':
                    priority_weights[container_id] = 10.0  # 10x weight for critical
                elif category == 'Urgent':
                    priority_weights[container_id] = 3.0   # 3x weight for urgent
                else:  # Safe
                    priority_weights[container_id] = 0.0   # Base weight for safe
                    
            self.logger.debug(
                f"Slack data found in container data:\n"
                f"  - Critical containers: {len(critical_containers)}\n"
                f"  - Weights assigned to {len(priority_weights)} containers"
            )
        else:
            # If no slack data, use default weight for all containers
            for container_id in self.container_data['container_id'].unique():
                priority_weights[container_id] = 1.0
                
            self.logger.debug("No slack data found in container data, using default weights")
        
        return has_slack_data, critical_containers, priority_weights

    def build_model(self, sequential: bool) -> None:
        """
        Build the optimization model with all variables,
        constraints, and objective function.
        """
        start_time = time.time()
        if not sequential:
            self.prepare_data()
        if self.model_data is None:
            raise ValueError("Data must be prepared before building model")
            
        self.logger.info("Building optimization model...")
        
        try:
            # Create Gurobi environment and model
            self.solver_env = gp.Env(params=self.gurobi_params)
            self.model = Model("TourFormation", env= self.solver_env)
            self.model.setParam("MIPGap",self.mip_gap)
            self.model.setParam("TimeLimit", self.time_limit)
            
            # Add variables
            self._add_variables()
            
            # Add constraints
            self._add_constraints()
            
            # Set objective
            self._set_objective()
            
            self.model.update()
            
            #self.model.write("tour_formation.lp")
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise
        
        finally:
            end_time = time.time()
            self.logger.info(f"Model building completed in {end_time - start_time:.2f} seconds")
    
    def cb(self, model, where):
        
        if where == GRB.Callback.MIPSOL:
            
            # Get model objective
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

            # Has objective changed?
            if abs(obj - model._cur_obj) > 1e-8:
                # If so, update incumbent and time
                model._cur_obj = obj
                model._time = time.time()

        # Terminate if objective has not improved in 20s
        if time.time() - model._time > self.early_termination_seconds:
            self.logger.info(f"Terminating: No improvement in {self.early_termination_seconds} seconds")
            model.terminate()
              
    def solve(self, sequential: bool) -> Dict[str, Any]:
        """
        Solve the optimization model and return solution.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing solution components
        """
        self.build_model(sequential)
        
        if self.model is None:
            raise ValueError("Model must be built before solving")
            
        try:
            self.count_constraints_by_type()
            self.logger.info("Solving optimization model...")
            start_time = time.time()
            
            self.model.setParam('DegenMoves', 4)
            self.model.setParam('GomoryPasses', 0)
            self.model.setParam('Threads', 16)
              
            #self._tune_parameters()                # dev mode - uncomment to run parameter tuning    
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            
            self.model.optimize(callback = self.cb)
            end_time = time.time()
            
            self.logger.info(f"Model solving completed in {end_time - start_time:.2f} seconds")
            
            # Check solution status
            if self.model.SolCount > 0:
                self.logger.info(f"Solution found with objective: {self.model.ObjVal}")
                self.solution = self._extract_solution()
                return self.solution
            elif self.model.status == GRB.INFEASIBLE:
                self._compute_and_log_iis(self.model)
            else:
                status_text = {
                    GRB.LOADED: "Model is loaded, but optimize() has not been called",
                    GRB.OPTIMAL: "Model was solved to optimality",
                    GRB.INFEASIBLE: "Model was proven to be infeasible",
                    GRB.INF_OR_UNBD: "Model was proven to be either infeasible or unbounded",
                    GRB.UNBOUNDED: "Model was proven to be unbounded",
                    GRB.CUTOFF: "Objective cutoff was reached",
                    GRB.ITERATION_LIMIT: "Iteration limit was reached",
                    GRB.NODE_LIMIT: "Node limit was reached",
                    GRB.TIME_LIMIT: "Time limit was reached",
                    GRB.SOLUTION_LIMIT: "Solution limit was reached",
                    GRB.INTERRUPTED: "Optimization was terminated by the user",
                    GRB.NUMERIC: "Optimization was terminated due to unrecoverable numerical difficulties",
                    GRB.SUBOPTIMAL: "Unable to satisfy optimality tolerances; returned solution is suboptimal",
                    GRB.INPROGRESS: "Optimization is in progress",
                    GRB.USER_OBJ_LIMIT: "User objective limit was reached"
                }.get(self.model.status, f"Unknown status: {self.model.status}")
                
                self.logger.warning(f"No optimal solution found. Status: {status_text}")
                
                # Return best solution found if available
                if self.model.SolCount > 0:
                    self.logger.info(f"Returning best solution found with objective: {self.model.ObjVal}")
                    self.solution = self._extract_solution()
                    return self.solution
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error solving model: {str(e)}")
            raise
            
    def _add_variables(self) -> None:
        """
        Add all decision variables to the model.
        Optimized to reduce variable count for single-location SKUs.
        """
        m = self.model
        data = self.model_data
        
        # 1. Tour activation variables (u_k)
        self.u = {}
        for k in data.tour_indices:
            self.u[k] = m.addVar(
                vtype=GRB.BINARY,
                name=f"u_{k}"
            )
        
        # 2. Container-Tour assignment variables (x_ik) 
        self.x = {}
        for i in data.container_ids:
            for k in data.tour_indices:
                self.x[i,k] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{i}_{k}"
                )
        
        # 3. Pick-Location assignment variables (y_isak) - ONLY for multi-location SKUs
        self.y = {}
        for i in data.container_ids:
            for s in data.multi_location_skus:
                if (i,s) in data.container_sku_qty:
                    required_qty = data.container_sku_qty[i,s]
                    for a in data.multi_location_skus[s]:
                        aisle_inventory = data.aisle_inventory.get((s,a), float('inf'))
                        max_qty = min(required_qty, aisle_inventory)
                        
                        for k in data.tour_indices:
                            self.y[i,s,a,k] = m.addVar(
                                vtype=GRB.INTEGER,
                                lb=0,
                                ub=max_qty,
                                name=f"y_{i}_{s}_{a}_{k}"
                            )
        
        # 4. Aisle range tracking variables - unchanged
        self.min_aisle = {}
        self.max_aisle = {}
        min_possible_aisle = min(
            min(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )
        max_possible_aisle = max(
            max(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )
        
        for k in data.tour_indices:
            self.min_aisle[k] = m.addVar(
                vtype=GRB.INTEGER,
                lb=min_possible_aisle,
                ub=max_possible_aisle,
                name=f"min_aisle_{k}"
            )
            self.max_aisle[k] = m.addVar(
                vtype=GRB.INTEGER,
                lb=min_possible_aisle,
                ub=max_possible_aisle,
                name=f"max_aisle_{k}"
            )
        
        # 5. Aisle visit indicator variables (z_isak) - ONLY for multi-location SKUs
        self.z = {}
        for i in data.container_ids:
            for s in data.multi_location_skus:
                if (i,s) in data.container_sku_qty:
                    for a in data.multi_location_skus[s]:
                        for k in data.tour_indices:
                            self.z[i,s,a,k] = m.addVar(
                                vtype=GRB.BINARY,
                                name=f"z_{i}_{s}_{a}_{k}"
                            )
        
        # 6. Aggregated Aisle Visit Variables (v_{a,k}) 
        self.v = {}
        for a in range(min_possible_aisle, max_possible_aisle + 1):
            for k in data.tour_indices:
                self.v[a, k] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"v_{a}_{k}"
                )

        m.update()
        
        # Variable statistics logging 
        total_vars = m.NumVars
        binary_vars = m.NumBinVars
        integer_vars = sum(1 for v in m.getVars() if v.VType == GRB.INTEGER)
        
        self.logger.info(f"Created {total_vars} variables:")
        self.logger.info(f"- Binary variables: {binary_vars}")
        self.logger.info(f"- Integer variables: {integer_vars}")
        self.logger.debug("Variable counts by type:")
        self.logger.debug(f"- Tour activation (u): {len(self.u)}")
        self.logger.debug(f"- Container-Tour assignment (x): {len(self.x)}")
        self.logger.debug(f"- Pick-Location assignment (y): {len(self.y)}")
        self.logger.debug(f"- Aisle range tracking (min/max): {len(self.min_aisle) + len(self.max_aisle)}")
        self.logger.debug(f"- Aisle visit indicator (z): {len(self.z)}")
        self.logger.debug(f"- Aisle-visit variables : {len(self.v)}")
        
    def _set_objective(self) -> None:
        """Objective function"""
        m = self.model
        data = self.model_data
        
        # Get priority weights based on slack
        _, _, priority_weights = self._get_slack_information()
    
        # 1. Lateness component (α) with priority weights
        self.lateness = gp.quicksum(
            priority_weights.get(i, 0.0) * self.x[i,k]
            for i in data.container_ids
            for k in data.tour_indices
        )
        
        # 2.1 Distinct aisles component - count of distinct aisles visited per tour
        self.distinct_aisles = gp.quicksum(
            self.v[a,k]
            for a, k in self.v.keys()
        )
        
        # 2.2 Aisle span component - max aisle minus min aisle
        self.aisle_span = gp.quicksum(
            (self.max_aisle[k] - self.min_aisle[k])
            for k in data.tour_indices
        )
        
        # 2.3 Combined travel distance
        self.travel_distance =  self.beta * self.distinct_aisles + self.aisle_span
        
        # 3. Tour count component (γ)
        self.tour_count = gp.quicksum(self.u[k] for k in data.tour_indices)
        
        # Set complete objective
        m.setObjective(
            - self.lateness + 
            self.travel_distance, #+
            #self.gamma * self.tour_count,
            GRB.MINIMIZE
        )
        
        m.update()  
          
    def _add_constraints(self) -> None:
        """Add all constraints to the model"""
        self._add_single_tour_assignment_constraints()
        self._add_tour_capacity_constraints()
        self._add_sku_fulfillment_constraints()
        self._add_inventory_limit_constraints()
        self._add_aisle_visit_linking_constraints()
        self._add_min_max_aisle_constraints()
        #self._add_critical_container_constraints()
        self._add_tour_ordering_constraints()
        self._add_valid_inequalities()
        
        # C1: Single-Tour Assignment    
    def _add_single_tour_assignment_constraints(self) -> None:
        """Add constraints ensuring each container is assigned to exactly one tour"""
        m = self.model
        data = self.model_data
        
        for i in data.container_ids:
            m.addConstr(
                gp.quicksum(self.x[i,k] for k in data.tour_indices) <= 1,
                name=f"singletour_{i}"
            )
            
    # C2 & C3: Tour Capacity Upper/Lower Bounds        
    def _add_tour_capacity_constraints(self) -> None:
        """
        Add constraints for tour capacity.
        All tours must be at full capacity except in final iteration.
        """
        m = self.model
        data = self.model_data
        
        if data.is_last_iteration:
            
            max_possible_tours = self.num_tours
            #if len(data.container_ids) % self.max_containers_per_tour != 0:
            #    max_possible_tours += 1
            m.addConstr(
                gp.quicksum(self.u[k] for k in data.tour_indices) == max_possible_tours,
                name="max_tours"
            )
            # Maximum capacity for all tours
            for k in data.tour_indices:
                m.addConstr(
                    gp.quicksum(self.x[i,k] for i in data.container_ids) <= 
                    self.max_containers_per_tour * self.u[k],
                    name=f"tourcapacityupper_{k}"
                )
                    
            remaining_containers = len(data.container_ids)
            for k in data.tour_indices:
                # If we have enough containers, enforce regular minimum
                if remaining_containers >= self.min_containers_per_tour:
                    m.addConstr(
                        gp.quicksum(self.x[i,k] for i in data.container_ids) >= 
                        self.min_containers_per_tour * self.u[k],
                        name=f"tourcapacitylower_{k}"
                    )
                    remaining_containers -= self.min_containers_per_tour
                # For the last tour(s), use remaining containers as minimum
                else:
                    m.addConstr(
                        gp.quicksum(self.x[i,k] for i in data.container_ids) >= 
                        remaining_containers * self.u[k],
                        name=f"tourcapacitylower_{k}"
                    )
        else: 
            max_possible_tours = self.num_tours
            m.addConstr(
                gp.quicksum(self.u[k] for k in data.tour_indices) == max_possible_tours,
                name="maxtours"
            )
            for k in data.tour_indices:

                # Force full capacity for non-final iterations
                m.addConstr(
                    gp.quicksum(self.x[i,k] for i in data.container_ids) == 
                    self.max_containers_per_tour * self.u[k],
                    name=f"tourcapacityfull_{k}"
                )
            
    def _add_sku_fulfillment_constraints(self) -> None:
        """
        Add constraints ensuring all required SKUs are picked.
        Optimized to handle single-location SKUs implicitly.
        """
        m = self.model
        data = self.model_data
        
        # For multi-location SKUs only - need to decide which aisles to pick from
        for i in data.container_ids:
            for s in data.multi_location_skus:
                if (i,s) in data.container_sku_qty:
                    for k in data.tour_indices:
                        m.addConstr(
                            gp.quicksum(self.y[i,s,a,k] for a in data.multi_location_skus[s]) == 
                            data.container_sku_qty[i,s] * self.x[i,k],
                            name=f"skufulfill_multi_{i}_{s}_{k}"
                        )
        
        # Single-location SKUs don't need fulfillment constraints since they only have one choice
        # The inventory constraints will ensure availability

    def _add_inventory_limit_constraints(self) -> None:
        """
        Add constraints for inventory availability.
        Handles both single-location and multi-location SKUs.
        """
        m = self.model
        data = self.model_data
        
        # For single-location SKUs
        for s in data.single_location_skus:
            a = data.single_location_skus[s]
            if (s,a) in data.aisle_inventory:
                m.addConstr(
                    gp.quicksum(
                        data.container_sku_qty[(i,s)] * self.x[i,k]
                        for i in data.container_ids 
                        for k in data.tour_indices
                        if (i,s) in data.container_sku_qty
                    ) <= data.aisle_inventory[(s,a)],
                    name=f"inventory_single_{s}_{a}"
                )
        
        # For multi-location SKUs
        for s in data.multi_location_skus:
            for a in data.multi_location_skus[s]:
                if (s,a) in data.aisle_inventory:
                    m.addConstr(
                        gp.quicksum(
                            self.y[i,s,a,k] 
                            for i in data.container_ids 
                            for k in data.tour_indices
                            if (i,s) in data.container_sku_qty and (i,s,a,k) in self.y
                        ) <= data.aisle_inventory[(s,a)],
                        name=f"inventory_multi_{s}_{a}"
                    )
                    
    # C6: Linking Pick Quantities with Aisle Visits               
    def _add_aisle_visit_linking_constraints(self) -> None:
        """
        Add constraints linking pick quantities to aisle visits.
        Optimized for single-location SKUs.
        """
        m = self.model
        data = self.model_data
        
        # Direct linking for single-location SKUs - if container is assigned to tour, fixed aisles must be visited
        for i in data.container_ids:
            fixed_aisles = data.container_fixed_aisles.get(i, {})
            for a in fixed_aisles:
                for k in data.tour_indices:
                    m.addConstr(
                        self.v[a, k] >= self.x[i, k],
                        name=f"fixed_aisle_{i}_{a}_{k}"
                    )
        
        # Regular constraints for multi-location SKUs
        for i in data.container_ids:
            for s in data.multi_location_skus:
                if (i,s) in data.container_sku_qty:
                    for a in data.multi_location_skus[s]:
                        for k in data.tour_indices:
                            m.addConstr(
                                self.y[i,s,a,k] <= 
                                data.container_sku_qty[i,s] * self.z[i,s,a,k],
                                name=f"linkyz_{i}_{s}_{a}_{k}"
                            )
                            m.addConstr(
                                self.v[a, k] >= self.z[i,s,a,k],
                                name=f"linkvz_{i}_{s}_{a}_{k}"
                            )
                            
    def _add_min_max_aisle_constraints(self) -> None:
        """
        Enforce aisle range constraints for tours.
        Optimized to handle single-location SKUs more efficiently.
        """
        m = self.model
        data = self.model_data

        # Determine the overall aisle range
        min_possible_aisle = min(
            min(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )
        max_possible_aisle = max(
            max(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )

        # Precompute potential min/max aisles for each container based on fixed aisles
        for i in data.container_ids:
            fixed_aisles = data.container_fixed_aisles.get(i, {})
            if fixed_aisles:
                min_fixed = min(fixed_aisles.keys())
                max_fixed = max(fixed_aisles.keys())
                
                for k in data.tour_indices:
                    # If container i is assigned to tour k, enforce its fixed aisle boundaries
                    m.addGenConstrIndicator(
                        self.x[i, k], 1,
                        self.min_aisle[k] <= min_fixed,
                        name=f"fixed_min_aisle_{i}_{k}"
                    )
                    m.addGenConstrIndicator(
                        self.x[i, k], 1,
                        self.max_aisle[k] >= max_fixed,
                        name=f"fixed_max_aisle_{i}_{k}"
                    )

        # Standard constraints for all aisles visited (via v[a,k])
        for k in data.tour_indices:
            for a in range(min_possible_aisle, max_possible_aisle + 1):
                m.addGenConstrIndicator(
                    self.v[a, k], 1,
                    self.min_aisle[k] <= a,
                    name=f"indminaisle_v_{a}_{k}"
                )
                m.addGenConstrIndicator(
                    self.v[a, k], 1,
                    self.max_aisle[k] >= a,
                    name=f"indmaxaisle_v_{a}_{k}"
                )
        
            # Ensure max_aisle is greater than min_aisle for active tours
            m.addConstr(
                self.max_aisle[k] >= self.min_aisle[k],
                name=f"aisle_order_{k}"
            )
    
    def _add_critical_container_constraints(self) -> None:
        """
        Add constraints ensuring all critical containers (with negative slack)
        are assigned to exactly one tour.
        """
        m = self.model
        data = self.model_data
        
        # Get slack information
        has_slack_data, critical_containers, _ = self._get_slack_information()
        
        if has_slack_data and critical_containers:
            # Only add constraints if we have critical containers
            critical_in_scope = set(data.container_ids).intersection(critical_containers)
            
            if critical_in_scope:
                self.logger.info(f"Adding assignment constraints for {len(critical_in_scope)} critical containers")
                
                for i in critical_in_scope:
                    m.addConstr(
                        gp.quicksum(self.x[i,k] for k in data.tour_indices) == 1,
                        name=f"critical_{i}"
                    )
                
    def _add_tour_ordering_constraints(self) -> None:
        """Enforce sequential use of tours"""
        m = self.model
        data = self.model_data
        
        for k in range(1, max(data.tour_indices) + 1):
            if k-1 in data.tour_indices and k in data.tour_indices:
                m.addConstr(
                    self.u[k] <= self.u[k-1],
                    name=f"tour_order_{k}"
                )
        
    def _add_valid_inequalities(self) -> None:
        m = self.model
        data = self.model_data
        
        # Determine the overall aisle range
        min_possible_aisle = min(
            min(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )
        max_possible_aisle = max(
            max(data.sku_aisles[s]) 
            for s in data.skus 
            if data.sku_aisles[s]
        )
        
        # 1. Minimum number of aisles per tour based on containers
        for k in data.tour_indices:
            m.addConstr(
                gp.quicksum(self.v[a, k] for a in range(min_possible_aisle, max_possible_aisle + 1)) >= 
                self.u[k],  # At least one aisle must be visited if tour is active
                name=f"min_aisles_{k}"
            )
        
        # 2. Aggregated container-aisle relationship
        for a in range(min_possible_aisle, max_possible_aisle + 1):
            containers_using_aisle = []
            for i in data.container_ids:
                if any(a in data.sku_aisles.get(s, []) for s in 
                    [s for s in data.skus if (i,s) in data.container_sku_qty]):
                    containers_using_aisle.append(i)
            
            if containers_using_aisle:
                for k in data.tour_indices:
                    m.addConstr(
                        self.v[a, k] <= gp.quicksum(self.x[i, k] for i in containers_using_aisle),
                        name=f"aisle_container_link_{a}_{k}"
                    )
        
        #3. Upper bound on distinct aisles per tour
        for k in data.tour_indices:
            unique_aisles_per_container = {}
            for i in data.container_ids:
                aisles = set()
                for s in data.skus:
                    if (i,s) in data.container_sku_qty:
                        aisles.update(data.sku_aisles.get(s, []))
                unique_aisles_per_container[i] = len(aisles)
            
            m.addConstr(
                gp.quicksum(self.v[a,k] for a in range(min_possible_aisle, max_possible_aisle + 1)) <= 
                gp.quicksum(unique_aisles_per_container[i] * self.x[i,k] for i in data.container_ids),
                name=f"tour_max_aisles_{k}"
            )
                          
    def _extract_solution(self) -> Dict[str, Any]:
        """
        Extract solution components from the optimized model.
        Handles both single-location and multi-location SKUs.
        """
        if self.model is None:
            raise ValueError("Model must be built before solving")
            
        try:
            if sum(self.u[k].X for k in self.model_data.tour_indices) > 0:
                lateness_value = self.lateness.getValue()
                distance_value = self.travel_distance.getValue()
                tour_count_value = self.tour_count.getValue()
                
                # Check if slack data is available
                has_slack_data, critical_containers, _ = self._get_slack_information()
                
                # Log detailed optimization results
                self.logger.info("Tour Formation Optimization Results:")
                self.logger.info("Component Values:")
                self.logger.info(f"  - Lateness: {lateness_value:.2f} hours")
                self.logger.info(f"  - Travel Distance (beta={self.beta}): {distance_value:.2f} aisles")
                self.logger.info(f"  - Tour Count (gamma={self.gamma}): {tour_count_value:.2f}")
                
                # Continue with standard solution extraction
                solution = {
                    'container_assignments': {},
                    'pick_assignments': {},
                    'aisle_ranges': {}
                }

                data = self.model_data

                # Extract container-tour assignments (unchanged)
                for i in data.container_ids:
                    for k in data.tour_indices:
                        if self.x[i,k].X > 0.5:
                            solution['container_assignments'][i] = {
                                'tour': k,
                                'lateness': max(0, (self.planning_timestamp - 
                                    self.container_data[
                                        self.container_data['container_id']==i
                                    ]['cut_datetime'].iloc[0]).total_seconds()/3600)
                            }

                # Extract pick assignments - Handle both types of SKUs
                for i in data.container_ids:
                    if i in solution['container_assignments']:
                        k = solution['container_assignments'][i]['tour']
                        
                        # 1. Extract single-location SKU picks (implicitly determined)
                        for s in data.single_location_skus:
                            if (i,s) in data.container_sku_qty:
                                a = data.single_location_skus[s]
                                if i not in solution['pick_assignments']:
                                    solution['pick_assignments'][i] = []
                                solution['pick_assignments'][i].append({
                                    'sku': s,
                                    'aisle': a,
                                    'qty': data.container_sku_qty[(i,s)],
                                    'tour': k
                                })
                        
                        # 2. Extract multi-location SKU picks (from y variables)
                        for s in data.multi_location_skus:
                            if (i,s) in data.container_sku_qty:
                                for a in data.multi_location_skus[s]:
                                    if (i,s,a,k) in self.y and self.y[i,s,a,k].X > 0:
                                        if i not in solution['pick_assignments']:
                                            solution['pick_assignments'][i] = []
                                        solution['pick_assignments'][i].append({
                                            'sku': s,
                                            'aisle': a,
                                            'qty': int(self.y[i,s,a,k].X),
                                            'tour': k
                                        })

                # Extract aisle ranges for active tours (unchanged)
                for k in data.tour_indices:
                    if self.u[k].X > 0.5: 
                        solution['aisle_ranges'][k] = {
                            'min_aisle': int(self.min_aisle[k].X),
                            'max_aisle': int(self.max_aisle[k].X)
                        }

                # Add summary metrics (unchanged)
                solution['metrics'] = {
                    'total_lateness': lateness_value,
                    'total_distance': distance_value,
                    'active_tours': tour_count_value,
                    'objective_value': self.model.objVal
                }

                return solution
            else:
                self.logger.warning(f"No optimal solution found. Status: {self.model.status}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error extracting solution: {str(e)}")
            raise
    
    def count_constraints_by_type(self) -> None:
        """
        Count and log the number of constraints by type in the optimization model.
        Should be called after model building but before solving.
        """
        if self.model is None:
            raise ValueError("Model must be built before counting constraints")
            
        # Initialize counters for each constraint type
        constraint_counts = {
            'singletour': 0,      # C1: Single tour assignment
            'tourcapacity': 0,    # C2: Tour capacity (both upper and lower)
            'skufulfill': 0,      # C3: SKU fulfillment
            'inventory': 0,       # C4: Inventory limit
            'linkyz': 0,          # C5: Linking pick quantities to aisle visits
            'indminaisle': 0,     # C6a: Min aisle indicator constraints
            'indmaxaisle': 0,     # C6b: Max aisle indicator constraints
            'aisleorder': 0,      # Aisle ordering (min ≤ max)
            'linkvz': 0,          # C6c: Linking aisle visits to v
            'critical': 0,        # C7: Critical container assignment
            'other': 0            # Any other constraints
        }
        
        # Count regular constraints
        for c in self.model.getConstrs():
            prefix = c.ConstrName.split('_')[0]
            if prefix in constraint_counts:
                constraint_counts[prefix] += 1
            else:
                constraint_counts['other'] += 1
                
        # Count indicator constraints
        for c in self.model.getGenConstrs():
            if c.GenConstrType == gp.GRB.GENCONSTR_INDICATOR:
                prefix = c.GenConstrName.split('_')[0]
                if prefix in constraint_counts:
                    constraint_counts[prefix] += 1
                else:
                    constraint_counts['other'] += 1
        
        # Get total number of constraints (regular + indicators)
        total_constraints = self.model.NumConstrs + sum(1 for _ in self.model.getGenConstrs())
        
        # Prepare summary table
        summary_table = []
        for constraint_type, count in constraint_counts.items():
            if count > 0:  # Only show constraint types that are present
                percentage = (count / total_constraints) * 100
                summary_table.append([
                    constraint_type.replace('_', ' ').title(),
                    count,
                    f"{percentage:.1f}%"
                ])
        
        # Add total row
        summary_table.append([
            "Total",
            total_constraints,
            "100.0%"
        ])
        
        # Log the summary
        self.logger.debug("\nConstraint Count Summary:")
        self.logger.debug("\n" + tabulate(
            summary_table,
            headers=['Constraint Type', 'Count', 'Percentage'],
            tablefmt='grid'
        ))
        
        return constraint_counts

    
    def generate_summary(self, solution: Dict[str, Any], available_containers: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate summary dataframes from solution.
        
        Parameters
        ----------
        solution : Dict[str, Any]
            Solution dictionary from solve method
        available_containers : pd.DataFrame
            Container data used in optimization
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing summary dataframes
        """
        # Get priority weights for lateness
        _, _, priority_weights = self._get_slack_information()
    
        container_assignments = []
        for container_id, assignment in solution['container_assignments'].items():
            container_data = {
                'ContainerID': container_id,
                'TourID': assignment['tour'],
                'ContainerCutTime': available_containers[
                    available_containers['container_id']==container_id
                ]['cut_datetime'].iloc[0],
                'Lateness': priority_weights.get(container_id, 1.0),  # Use priority weight as lateness,
                'MinAisle': solution['aisle_ranges'][assignment['tour']]['min_aisle'],
                'MaxAisle': solution['aisle_ranges'][assignment['tour']]['max_aisle']
            }
            container_assignments.append(container_data)
        
        container_tour_df = pd.DataFrame(container_assignments)

        # Create pick assignments dataframe
        pick_assignments = []
        for container_id, picks in solution['pick_assignments'].items():
            tour_id = solution['container_assignments'][container_id]['tour']
            for pick in picks:
                pick_assignments.append({
                    'ContainerID': container_id,
                    'TourID': tour_id,
                    'SKU': pick['sku'],
                    'Aisle': pick['aisle'],
                    'Quantity': pick['qty']
                })
        
        pick_assignments_df = pd.DataFrame(pick_assignments)

        # Generate tour metrics
        tour_metrics = []
        for tour_id in sorted(solution['aisle_ranges'].keys()):
            tour_containers = container_tour_df[container_tour_df['TourID'] == tour_id]
            tour_picks = pick_assignments_df[pick_assignments_df['TourID'] == tour_id]
            
            tour_metrics.append({
                'TourID': tour_id,
                'ContainerCount': len(tour_containers),
                'TotalQuantity': tour_picks['Quantity'].sum(),
                'UniqueAisles': len(tour_picks['Aisle'].unique()),
                'UniqueSKUs': len(tour_picks['SKU'].unique()),
                'TotalLateness': tour_containers['Lateness'].sum(),
                'MinAisle': solution['aisle_ranges'][tour_id]['min_aisle'],
                'MaxAisle': solution['aisle_ranges'][tour_id]['max_aisle'],
                'AisleRange': f"{solution['aisle_ranges'][tour_id]['min_aisle']}-{solution['aisle_ranges'][tour_id]['max_aisle']}",
                'AisleSpan': solution['aisle_ranges'][tour_id]['max_aisle'] - solution['aisle_ranges'][tour_id]['min_aisle']
            })

        tour_metrics_df = pd.DataFrame(tour_metrics)
        
        # Create and log tour summary table
        summary_table = []
        active_tours = tour_metrics_df[tour_metrics_df['ContainerCount'] > 0]
        for _, tour in active_tours.iterrows():
            summary_table.append([
                f"Tour {tour['TourID']}",
                tour['ContainerCount'], 
                tour['UniqueSKUs'],
                tour['UniqueAisles'],
                str(tour['AisleRange']),  
                tour['AisleSpan'],
                f"{tour['TotalQuantity']:,}",
                f"{tour['TotalLateness']:.1f}"
            ])
        
        headers = ['Tour', 'Containers', 'SKUs', 'Aisles', 'Aisle Range', 'Aisle Span', 'Total Units', 'Lateness (hrs)']
        self.logger.info("\nTour Summary:")
        self.logger.info("\n" + tabulate(summary_table, headers=headers, tablefmt='grid'))
        
        # Print overall summary
        total_tours = len(active_tours)
        sum_unique_aisles = active_tours['UniqueAisles'].sum()
        sum_aisle_span = active_tours['AisleSpan'].sum()
        sum_units = active_tours['TotalQuantity'].sum()
        sum_lateness = active_tours['TotalLateness'].sum()
        
        self.logger.info(
            f"\nOverall Summary:\n"
            f"- Total Tours: {total_tours}\n"
            f"- Sum of Unique Aisles: {sum_unique_aisles}\n"
            f"- Sum of Aisle Span: {sum_aisle_span}\n"
            f"- Sum of Units: {sum_units}\n"
            f"- Sum of Lateness: {sum_lateness:.1f} hours"
        )
        
        return {
            'container_assignments': container_tour_df,
            'pick_assignments': pick_assignments_df,
            'tour_metrics': tour_metrics_df
        }
        
    def _compute_and_log_iis(self, model: gp.Model):
        """
        Compute and log the Irreducible Inconsistent Subsystem (IIS) grouped by constraint types.
        """
        print("Computing IIS to identify problematic constraints...")
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
                self.logger.debug(f'\t{c.ConstrName}: {model.getRow(c)} {c.Sense} {c.RHS}')     
               
                constraint_type = c.ConstrName.split('_')[0]
                constraint_groups[constraint_type] = constraint_groups.get(constraint_type, 0) + 1
        
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

    def _tune_parameters(self) -> None:
        """
        Perform automated parameter tuning using Gurobi's tuning tool.
        Stores the best parameter settings found during the tuning process.
        """
        if self.model is None:
            raise ValueError("Model must be built before tuning parameters")
            
        try:
            self.logger.info("Starting Gurobi parameter tuning...")
            
            self.model.setParam('TuneTimeLimit', 600)
            
            # Run tuning process
            self.model.tune()
            
            for i in range( self.model.tuneResultCount):
                self.model.getTuneResult(i)
                self.model.write('tune'+str(i)+'.prm')
                
        except Exception as e:
            self.logger.error(f"Error during parameter tuning: {str(e)}")
            raise
    
    @classmethod
    def solve_sequential(cls, container_data: pd.DataFrame, slotbook_data: pd.DataFrame, 
                        planning_timestamp: datetime, config: Dict[str, Any],
                        generate_visuals: bool = False) -> Dict[str, Any]:
        """
        Solve the tour formation problem using sequential clustering approach.
        Handles unassigned containers by recursively clustering them and iteratively
        processing residuals until all containers are assigned or no further progress is made.
        """
        # Get configuration parameters
        containers_per_tour = config['tour_formation']['max_containers_per_tour']
        max_cluster_size = config.get('tour_formation', {}).get('max_cluster_size', 200)
        max_picking_capacity = config['global']['hourly_container_target']
        
        # Setup logging
        logger = setup_logging(config, 'sequential_tour_formation')
        logger.info(f"Starting sequential tour formation with max cluster size {max_cluster_size}")
        
        # Initialize combined solution
        combined_solution = {
            'container_assignments': {},
            'pick_assignments': {},
            'aisle_ranges': {},
            'metrics': {
                'total_lateness': 0.0,
                'total_distance': 0.0,
                'active_tours': 0
            }
        }
        
        # Track all containers and processed containers
        all_container_ids = set(container_data['container_id'].unique())
        processed_containers = set()
        tour_id_offset = 0
        
        # Check if prioritize_critical flag is set
        prioritize_critical = config.get('tour_formation', {}).get('prioritize_critical', True)
        
        if len(all_container_ids) > max_cluster_size:
            # Use our enhanced clustering algorithm
            clusters_result  = modified_cluster_containers(
                container_data, 
                slotbook_data, 
                max_cluster_size=max_cluster_size,
                containers_per_tour=containers_per_tour,
                use_distinct_aisles=False,
                generate_visuals=generate_visuals,
                output_path=config.get('visualization_path', './cluster_analysis'),
                prioritize_critical=prioritize_critical,
                max_picking_capacity = max_picking_capacity
            )
        else:
            # If total containers are within max_cluster_size, create a single cluster
            clusters = {0: list(all_container_ids)}
            # Create a simple stats DataFrame for the single cluster
            cluster_stats_df = pd.DataFrame([{
                'ClusterID': '0', 
                'TotalContainers': len(all_container_ids),
                'NumTours': max(1, len(all_container_ids) // containers_per_tour)
            }])
        
        clusters, cluster_stats_df = clusters_result
        
        # Process each cluster sequentially
        for cluster_id, cluster_container_ids in clusters.items():
            # Skip empty clusters
            if not cluster_container_ids:
                continue
                
            # Filter out already processed containers
            cluster_container_ids = [c_id for c_id in cluster_container_ids if c_id not in processed_containers]
        
            if not cluster_container_ids or len(cluster_container_ids) < config['tour_formation']['min_containers_per_tour']:
                continue
                
            logger.info(f"Processing cluster {cluster_id} with {len(cluster_container_ids)} containers")
            
            # Get the number of tours for this cluster from the stats DataFrame
            cluster_stats_row = cluster_stats_df[cluster_stats_df['ClusterID'] == str(cluster_id)]
            num_tours = 1  # Default to 1 if not found
            
            if not cluster_stats_row.empty and 'NumTours' in cluster_stats_row.columns:
                num_tours = int(cluster_stats_row['NumTours'].iloc[0])
                logger.info(f"Cluster {cluster_id} requires {num_tours} tours based on statistics")
            else:
                logger.warning(f"No tour count found for cluster {cluster_id}, using default value of 1")
            
            # Create a solver instance for this cluster with the number of tours
            solver = cls(
                container_data=container_data,
                slotbook_data=slotbook_data,
                planning_timestamp=planning_timestamp,
                config=config,
                num_tours=num_tours  # Pass the tour count as a parameter
            )
            
            # Prepare data for this specific cluster
            solver.prepare_data(container_ids=cluster_container_ids)
            
            # Set the tour ID offset
            solver.tour_id_offset = tour_id_offset
            
            # Solve for this cluster
            cluster_solution = solver.solve(sequential=True)
            
            if cluster_solution:
                # Adjust tour IDs using the offset
                adjusted_solution = cls._adjust_tour_ids(cluster_solution, tour_id_offset)
                
                # Update the offset for the next cluster
                max_tour_id = max(adjusted_solution['aisle_ranges'].keys()) if adjusted_solution['aisle_ranges'] else tour_id_offset
                tour_id_offset = max_tour_id + 1
                
                # Merge solutions
                cls._merge_solutions(combined_solution, adjusted_solution)
                
                # Update processed containers
                processed_containers.update(adjusted_solution['container_assignments'].keys())
                
                logger.info(f"Cluster {cluster_id} solution: {len(adjusted_solution['container_assignments'])} containers assigned")
            else:
                logger.warning(f"No solution found for cluster {cluster_id}")
        
        # Check if any containers remain unassigned after all attempts
        final_unassigned = all_container_ids - processed_containers
        if final_unassigned:
            logger.warning(f"{len(final_unassigned)} containers remain unassigned")
        
        # Log final statistics
        logger.info(f"Sequential optimization complete: {len(combined_solution['container_assignments'])} out of {len(all_container_ids)} containers assigned to {combined_solution['metrics']['active_tours']} tours")
        
        return combined_solution


    @staticmethod
    def _adjust_tour_ids(solution: Dict[str, Any], offset: int) -> Dict[str, Any]:
        """Adjust tour IDs in a solution by adding an offset"""
        if offset == 0:
            return solution  # No adjustment needed
            
        adjusted = {
            'container_assignments': {},
            'pick_assignments': {},
            'aisle_ranges': {},
            'metrics': solution['metrics'].copy()
        }
        
        # Adjust container assignments
        for container_id, assignment in solution['container_assignments'].items():
            adjusted['container_assignments'][container_id] = {
                'tour': assignment['tour'] + offset,
                'lateness': assignment['lateness']
            }
        
        # Adjust pick assignments
        for container_id, picks in solution['pick_assignments'].items():
            adjusted['pick_assignments'][container_id] = []
            for pick in picks:
                adjusted_pick = pick.copy()
                adjusted_pick['tour'] = pick['tour'] + offset
                adjusted['pick_assignments'][container_id].append(adjusted_pick)
        
        # Adjust aisle ranges
        for tour_id, aisle_range in solution['aisle_ranges'].items():
            adjusted['aisle_ranges'][tour_id + offset] = aisle_range.copy()
        
        return adjusted

    @staticmethod
    def _merge_solutions(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge source solution into target solution"""
        # Merge container assignments
        target['container_assignments'].update(source['container_assignments'])
        
        # Merge pick assignments
        for container_id, picks in source['pick_assignments'].items():
            target['pick_assignments'][container_id] = picks
        
        # Merge aisle ranges
        target['aisle_ranges'].update(source['aisle_ranges'])
        
        # Update metrics
        target['metrics']['total_lateness'] += source['metrics']['total_lateness']
        target['metrics']['total_distance'] += source['metrics']['total_distance']
        target['metrics']['active_tours'] += source['metrics']['active_tours']