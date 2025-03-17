"""
Tour Formation Model module.

This module handles the Gurobi model building, variables, constraints, and objective function
definition for the tour formation optimization problem.
"""

import gurobipy as gp
from gurobipy import GRB, Model
import time
from typing import Dict, Any, Callable, Optional
from tabulate import tabulate
import logging

class TourFormationModel:
    """
    Optimization model for the tour formation problem.
    
    Handles the building of the Gurobi model, including variables, constraints,
    and objective function definitions.
    """
    
    def __init__(self, config: Dict[str, Any], model_data, logger: logging.Logger):
        """
        Initialize the tour formation model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        model_data : ModelData
            Data object containing preprocessed model data
        logger : logging.Logger
            Logger instance
        """
        self.config = config
        self.model_data = model_data
        self.logger = logger
        
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
        
        # Gurobi configs
        self.output_flag = pick_config['solver']['output_flag']
        self.mip_gap = pick_config['solver']['mip_gap']
        self.time_limit = pick_config['solver']['time_limit']

        # Initialize Gurobi parameters
        gurobi_config = self.config.get('gurobi', {})
        self.gurobi_params = {
            "OutputFlag": self.output_flag,
            "GURO_PAR_ISVNAME": gurobi_config.get('ISV_NAME'),
            "GURO_PAR_ISVAPPNAME": gurobi_config.get('APP_NAME'),
            "GURO_PAR_ISVEXPIRATION": gurobi_config.get('EXPIRATION'),
            "GURO_PAR_ISVKEY": gurobi_config.get('CODE')
        }
        
        # Initialize model and variables
        self.model = None
        self.solver_env = None
        
        # Decision variables
        self.u = {}  # Tour activation variables (u_k)
        self.x = {}  # Container-Tour assignment variables (x_ik)
        self.y = {}  # Pick-Location assignment variables (y_isak) - for multi-location SKUs
        self.min_aisle = {}  # Min aisle for tour k
        self.max_aisle = {}  # Max aisle for tour k
        self.z = {}  # Aisle visit indicator variables (z_isak) - for multi-location SKUs
        self.v = {}  # Aggregated Aisle Visit Variables (v_{a,k})
        
        # Objective components
        self.lateness = None
        self.distinct_aisles = None
        self.aisle_span = None
        self.travel_distance = None
        self.tour_count = None

    def build_model(self, sequential: bool = False) -> None:
        """
        Build the optimization model with all variables,
        constraints, and objective function.
        
        Parameters
        ----------
        sequential : bool, optional
            Whether this is part of a sequential solving process, by default False
        """
        start_time = time.time()
        self.logger.info("Building optimization model...")
        
        try:
            # Create Gurobi environment and model
            self.solver_env = gp.Env(params=self.gurobi_params)
            self.model = Model("TourFormation", env=self.solver_env)
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
        
        finally:
            end_time = time.time()
            self.logger.info(f"Model building completed in {end_time - start_time:.2f} seconds")
    
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
        
        self.logger.debug(f"Created {total_vars} variables:")
        self.logger.debug(f"- Binary variables: {binary_vars}")
        self.logger.debug(f"- Integer variables: {integer_vars}")
        self.logger.debug("Variable counts by type:")
        self.logger.debug(f"- Tour activation (u): {len(self.u)}")
        self.logger.debug(f"- Container-Tour assignment (x): {len(self.x)}")
        self.logger.debug(f"- Pick-Location assignment (y): {len(self.y)}")
        self.logger.debug(f"- Aisle range tracking (min/max): {len(self.min_aisle) + len(self.max_aisle)}")
        self.logger.debug(f"- Aisle visit indicator (z): {len(self.z)}")
        self.logger.debug(f"- Aisle-visit variables : {len(self.v)}")
    
    def _add_constraints(self) -> None:
        """Add all constraints to the model"""
        self._add_single_tour_assignment_constraints()
        self._add_tour_capacity_constraints()
        self._add_sku_fulfillment_constraints()
        self._add_inventory_limit_constraints()
        self._add_aisle_visit_linking_constraints()
        self._add_min_max_aisle_constraints()
        self._add_tour_ordering_constraints()
        self._add_valid_inequalities()
    
    def _add_single_tour_assignment_constraints(self) -> None:
        """Add constraints ensuring each container is assigned to exactly one tour"""
        m = self.model
        data = self.model_data
        
        for i in data.container_ids:
            m.addConstr(
                gp.quicksum(self.x[i,k] for k in data.tour_indices) <= 1,
                name=f"singletour_{i}"
            )
            
    def _add_tour_capacity_constraints(self) -> None:
        """
        Add constraints for tour capacity.
        All tours must be at full capacity except in final iteration.
        """
        m = self.model
        data = self.model_data
        
        if data.is_last_iteration:
            
            max_possible_tours = len(data.tour_indices)
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
            max_possible_tours = len(data.tour_indices)
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
        
        # Get slack information - assuming this is passed in model_data
        critical_containers = getattr(data, 'critical_containers', set())
        
        if critical_containers:
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
        """Add valid inequalities to strengthen the formulation"""
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
        
        # 3. Upper bound on distinct aisles per tour
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
            
    def _set_objective(self) -> None:
        """Objective function definition"""
        m = self.model
        data = self.model_data
        
        # Get priority weights based on slack
        priority_weights = getattr(data, 'priority_weights', {})
        # Default to 1.0 if not found
        for i in data.container_ids:
            if i not in priority_weights:
                priority_weights[i] = 1.0
    
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
        self.travel_distance = self.beta * self.distinct_aisles + self.aisle_span
        
        # 3. Tour count component (γ)
        self.tour_count = gp.quicksum(self.u[k] for k in data.tour_indices)
        
        # Set complete objective
        m.setObjective(
            - self.lateness + 
            self.travel_distance, # +
            # self.gamma * self.tour_count,
            GRB.MINIMIZE
        )
        
        m.update()
        
    def get_callback(self) -> Callable:
        """
        Get the optimization callback function.
        
        Returns
        -------
        Callable
            Callback function for early termination
        """
        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                # Get model objective
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

                # Has objective changed?
                if abs(obj - model._cur_obj) > 1e-8:
                    # If so, update incumbent and time
                    model._cur_obj = obj
                    model._time = time.time()

            # Terminate if objective has not improved in specified time
            if time.time() - model._time > self.early_termination_seconds:
                self.logger.debug(f"Terminating: No improvement in {self.early_termination_seconds} seconds")
                model.terminate()
                self.model._terminated_early = True
        
        return cb

    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Solve the optimization model and return solution.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Solution if found, None otherwise
        """
        if self.model is None:
            raise ValueError("Model must be built before solving")
            
        try:
            self.logger.info("Solving optimization model...")
            start_time = time.time()
            
            # Initialize callback
            self.model._cur_obj = float('inf')
            self.model._time = time.time()
            self.model._terminated_early = True
            
            # Optimize with callback
            self.model.optimize(self.get_callback())
            
            # Check solution status
            if self.model.SolCount > 0:
                if self.model._terminated_early:
                    self.logger.info(f"Solution terminated early due to no improvement in {self.early_termination_seconds} seconds")
                    self.logger.info(f"Solution found with objective: {self.model.ObjVal}")
                    self.logger.info(f"Solution found in {time.time() - start_time:.2f} seconds")
                    return self._extract_solution()
            else:
                self.logger.warning(f"No optimal solution found. Status: {self.model.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error solving model: {str(e)}")
            raise

    def _extract_solution(self) -> Dict[str, Any]:
        """
        Extract solution components from the optimized model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing solution components
        """
        try:
            if sum(self.u[k].X for k in self.model_data.tour_indices) > 0:
                # Get objective component values
                lateness_value = self.lateness.getValue()
                distance_value = self.travel_distance.getValue()
                tour_count_value = self.tour_count.getValue()
                
                # Log detailed optimization results
                self.logger.info("Tour Formation Optimization Results:")
                self.logger.info("Component Values:")
                self.logger.info(f"  - Slack: {lateness_value:.2f}")
                self.logger.info(f"  - Travel Distance (beta={self.beta}): {distance_value:.2f}")
                self.logger.info(f"  - Tour Count (gamma={self.gamma}): {tour_count_value:.2f}")
                
                # Initialize solution dictionary
                solution = {
                    'container_assignments': {},
                    'pick_assignments': {},
                    'aisle_ranges': {},
                    'metrics': {}
                }

                data = self.model_data

                # Extract container-tour assignments
                for i in data.container_ids:
                    for k in data.tour_indices:
                        if self.x[i,k].X > 0.5:
                            solution['container_assignments'][i] = {
                                'tour': k
                            }

                # Extract pick assignments for multi-location SKUs
                for i in data.container_ids:
                    picks = []
                    for s in data.multi_location_skus:
                        if (i,s) in data.container_sku_qty:
                            for a in data.multi_location_skus[s]:
                                for k in data.tour_indices:
                                    if (i,s,a,k) in self.y and self.y[i,s,a,k].X > 0:
                                        picks.append({
                                            'sku': s,
                                            'aisle': a,
                                            'quantity': int(self.y[i,s,a,k].X)
                                        })
                    if picks:
                        solution['pick_assignments'][i] = picks

                # Extract aisle ranges for active tours
                for k in data.tour_indices:
                    if self.u[k].X > 0.5:
                        solution['aisle_ranges'][k] = {
                            'min_aisle': int(self.min_aisle[k].X),
                            'max_aisle': int(self.max_aisle[k].X)
                        }

                # Add metrics
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

    def count_constraints_by_type(self) -> Dict[str, int]:
        """
        Count and log the number of constraints by type in the optimization model.
        Should be called after model building but before solving.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping constraint types to counts
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
    
    def compute_and_log_iis(self, model: gp.Model) -> None:
        """
        Compute and log the Irreducible Inconsistent Subsystem (IIS) grouped by constraint types.
        
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
    
    def tune_parameters(self) -> None:
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
            
            for i in range(self.model.tuneResultCount):
                self.model.getTuneResult(i)
                self.model.write(f'tune{i}.prm')
                
        except Exception as e:
            self.logger.error(f"Error during parameter tuning: {str(e)}")
            raise
