import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
import math

CONFIG = {
    'max_volume_per_tour': 40,  # Maximum volume per tour (picking capacity)
    'min_volume_per_tour': 20,  # Minimum volume per tour (picking capacity)
    'travel_distance_weight': 1.0, #Travel weight for objective function
    'new_tour_weight': 10.0,  # Setup and changeover weight for a new tour
    'params': {
                "GURO_PAR_ISVNAME": "Chewy",
                "GURO_PAR_ISVAPPNAME": "DS",
                "GURO_PAR_ISVEXPIRATION": 20260331,
                "GURO_PAR_ISVKEY": "XG5VKFHH",
                'mip_gap': 0.01,
                'time_limit': 300,
                'output_flag': 0
                }
}

@dataclass
class ModelData:
    container_ids: List[str]
    skus: List[str]
    container_sku_qty: Dict[Tuple[str, str], int]
    container_volumes: Dict[str, float]  # Container ID -> volume
    sku_aisles: Dict[str, List[int]]
    aisle_inventory: Dict[Tuple[str, int], int]
    tour_indices: List[int]
    max_aisle: int
    multi_location_skus: Dict[str, List[int]]
    single_location_skus: Dict[str, int]  # SKU -> aisle
    container_fixed_aisles: Dict[str, set]  # Container -> set of fixed aisles
    critical_containers: set  # Set of critical container IDs
    slack_weights: Dict[str, float]  # Container -> slack weight

def generate_sample_data(num_containers: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate container IDs
    container_ids = [f"C{i+1}" for i in range(num_containers)]
    
    # Generate SKUs
    num_skus = max(3, num_containers)  # At least 3 SKUs
    skus = [f"S{j+1}" for j in range(num_skus)]
    
    # Assign each container a random SKU and quantity
    sample_containers = pd.DataFrame({
        'container_id': container_ids,
        'item_number': np.random.choice(skus, num_containers),
        'pick_quantity': np.random.randint(1, 3, num_containers),
        'volume': np.random.randint(5, 10, num_containers)
    })
    
    # Assign each SKU a random aisle and inventory
    aisle_sequences = np.random.randint(1, 10, num_skus)
    inventory_qty = np.random.randint(50, 200, num_skus)
    sample_slotbook = pd.DataFrame({
        'item_number': skus,
        'aisle_sequence': aisle_sequences,
        'inventory_qty': inventory_qty
    })
    
    return sample_containers, sample_slotbook

# Example usage: set number of containers and seed
num_containers = 20
seed = 123
sample_containers, sample_slotbook = generate_sample_data(num_containers, seed)

# Identify multi-location SKUs from sample_slotbook
multi_location_skus = {}
sku_aisle_counts = sample_slotbook.groupby('item_number')['aisle_sequence'].nunique()
for sku, count in sku_aisle_counts.items():
    if count > 1:
        aisles = sample_slotbook[sample_slotbook['item_number'] == sku]['aisle_sequence'].tolist()
        multi_location_skus[sku] = aisles

# ============================================================================
# GUROBI MODEL IMPLEMENTATION
# ============================================================================

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Extract data from sample DataFrames
container_ids = sample_containers['container_id'].unique().tolist()
skus = sample_containers['item_number'].unique().tolist()

# Container-SKU quantity mapping
container_sku_qty = {}
for _, row in sample_containers.iterrows():
    container_sku_qty[(row['container_id'], row['item_number'])] = row['pick_quantity']

# Container volume mapping - fixed to use actual volume data
container_volumes = {}
for _, row in sample_containers.iterrows():
    container_volumes[row['container_id']] = row['volume']

# SKU-Aisle mapping
sku_aisles = {}
for _, row in sample_slotbook.iterrows():
    sku = row['item_number']
    aisle = row['aisle_sequence']
    if sku not in sku_aisles:
        sku_aisles[sku] = []
    sku_aisles[sku].append(aisle)

# Aisle inventory
aisle_inventory = {}
for _, row in sample_slotbook.iterrows():
    aisle_inventory[(row['item_number'], row['aisle_sequence'])] = row['inventory_qty']

# Single location SKUs (SKUs that only appear in one aisle)
single_location_skus = {}
for sku in skus:
    if sku in sku_aisles and len(sku_aisles[sku]) == 1:
        single_location_skus[sku] = sku_aisles[sku][0]

# Container fixed aisles (aisles that must be visited for each container)
container_fixed_aisles = {}
for container_id in container_ids:
    fixed_aisles = set()
    for sku in skus:
        if (container_id, sku) in container_sku_qty:
            if sku in sku_aisles:
                fixed_aisles.update(sku_aisles[sku])
    container_fixed_aisles[container_id] = fixed_aisles

# Tour indices (simple calculation based on total volume)
total_volume = sum(container_volumes.values())
min_tours = math.ceil(1.2*total_volume / CONFIG['max_volume_per_tour'] + 0.01)  # add 20% buffer
tour_indices = list(range(min_tours))

# Max aisle
max_aisle = sample_slotbook['aisle_sequence'].max()

#TODO: Changes here
# Generate slack weights for each container id randomly choosing from 5, 10, or 15
slack_weights = {cid: random.choice([5, 5, 5]) for cid in container_ids}

critical_containers = [cid for cid, w in slack_weights.items() if w == 10]

# Add slack_weights and critical_container flag to sample_containers
sample_containers['slack_weight'] = sample_containers['container_id'].map(slack_weights)
sample_containers['critical_container'] = sample_containers['container_id'].isin(critical_containers)

sample_containers.to_csv("sample_containers.csv", index=False)
sample_slotbook.to_csv("sample_slotbook.csv", index=False)

# Create model data
model_data = ModelData(
    container_ids=container_ids,
    skus=skus,
    container_sku_qty=container_sku_qty,
    container_volumes=container_volumes,
    sku_aisles=sku_aisles,
    aisle_inventory=aisle_inventory,
    tour_indices=tour_indices,
    max_aisle=max_aisle,
    multi_location_skus=multi_location_skus,
    single_location_skus=single_location_skus,
    container_fixed_aisles=container_fixed_aisles,
    critical_containers=critical_containers,
    slack_weights=slack_weights
)

# ============================================================================
# BUILD GUROBI MODEL
# ============================================================================

# Create model
env = gp.Env(params=CONFIG['params'])
model = gp.Model("TourFormation", env=env)

# ============================================================================
# VARIABLES
# ============================================================================

# 1. Tour activation variables (u_k)
u = {}
for k in model_data.tour_indices:
    u[k] = model.addVar(vtype=GRB.BINARY,name=f"u_{k}")

# 2. Container-Tour assignment variables (x_ik) 
x = {}
for i in model_data.container_ids:
    for k in model_data.tour_indices:
        x[i,k] = model.addVar(vtype=GRB.BINARY,name=f"x_{i}_{k}")

# 3. Pick-Location assignment variables (y_isak) - ONLY for multi-location SKUs
y = {}
for i in model_data.container_ids:
    for s in model_data.multi_location_skus:
        if (i,s) in model_data.container_sku_qty:
            required_qty = model_data.container_sku_qty[i,s]
            for a in model_data.multi_location_skus[s]:
                aisle_inventory = model_data.aisle_inventory.get((s,a), float('inf'))
                max_qty = min(required_qty, aisle_inventory)
                
                for k in model_data.tour_indices:
                    y[i,s,a,k] = model.addVar(vtype=GRB.INTEGER,lb=0,ub=max_qty,name=f"y_{i}_{s}_{a}_{k}")

# 4. Aisle range tracking variables - fixed variable name conflicts
min_aisle_var = {}
max_aisle_var = {}
min_possible_aisle = min(min(model_data.sku_aisles[s]) for s in model_data.skus if model_data.sku_aisles[s])
max_possible_aisle = max(max(model_data.sku_aisles[s]) for s in model_data.skus if model_data.sku_aisles[s])

for k in model_data.tour_indices:
    min_aisle_var[k] = model.addVar(vtype=GRB.INTEGER,lb=min_possible_aisle,ub=max_possible_aisle,name=f"min_aisle_{k}")
    max_aisle_var[k] = model.addVar(vtype=GRB.INTEGER,lb=min_possible_aisle,ub=max_possible_aisle,name=f"max_aisle_{k}")

# 5. Aisle visit indicator variables (z_isak) - ONLY for multi-location SKUs
z = {}
for i in model_data.container_ids:
    for s in model_data.multi_location_skus:
        if (i,s) in model_data.container_sku_qty:
            for a in model_data.multi_location_skus[s]:
                for k in model_data.tour_indices:
                    z[i,s,a,k] = model.addVar(vtype=GRB.BINARY,name=f"z_{i}_{s}_{a}_{k}")

# 6. Aggregated Aisle Visit Variables (v_{a,k}) 
v = {}
for a in range(min_possible_aisle, max_possible_aisle + 1):
    for k in model_data.tour_indices:
        v[a, k] = model.addVar(vtype=GRB.BINARY,name=f"v_{a}_{k}")

model.update()

# ============================================================================
# CONSTRAINTS
# ============================================================================

#TODO: Changes here
# _add_single_tour_assignment_constraints
for i in model_data.container_ids:
            model.addConstr(gp.quicksum(x[i,k] for k in model_data.tour_indices) <= 1, name=f"singletour_{i}")
            #model.addConstr(gp.quicksum(x[i,k] for k in model_data.tour_indices) == 1, name=f"singletour_{i}")

# _add_tour_capacity_constraints
for k in model_data.tour_indices:
    model.addConstr(
        gp.quicksum(x[i,k] * model_data.container_volumes[i] for i in model_data.container_ids) 
        <= CONFIG['max_volume_per_tour'] * u[k], name=f"tourcapacityupper_{k}")

#TODO: Changes here
# _add_tour_capacity_constraints
for k in model_data.tour_indices:
    model.addConstr(
        gp.quicksum(x[i,k] * model_data.container_volumes[i] for i in model_data.container_ids) 
        >= CONFIG['min_volume_per_tour'] * u[k], name=f"tourcapacitylower_{k}")
    
# _add_sku_fulfillment_constraints
for i in model_data.container_ids:
    for s in model_data.multi_location_skus:
        if (i,s) in model_data.container_sku_qty:
            for k in model_data.tour_indices:
                model.addConstr(
                    gp.quicksum(y[i,s,a,k] for a in model_data.multi_location_skus[s]) == 
                    model_data.container_sku_qty[i,s] * x[i,k],name=f"skufulfill_multi_{i}_{s}_{k}")

# _add_inventory_limit_constraints
for s in model_data.single_location_skus:
    a = model_data.single_location_skus[s]
    if (s,a) in model_data.aisle_inventory:
        model.addConstr(
            gp.quicksum(
                model_data.container_sku_qty[(i,s)] * x[i,k]
                for i in model_data.container_ids 
                for k in model_data.tour_indices
                if (i,s) in model_data.container_sku_qty
            ) <= model_data.aisle_inventory[(s,a)],
            name=f"inventory_single_{s}_{a}"
            )

# For multi-location SKUs
for s in model_data.multi_location_skus:
    for a in model_data.multi_location_skus[s]:
        if (s,a) in model_data.aisle_inventory:
            model.addConstr(
                gp.quicksum(
                    y[i,s,a,k] 
                    for i in model_data.container_ids 
                    for k in model_data.tour_indices
                    if (i,s) in model_data.container_sku_qty and (i,s,a,k) in y
                ) <= model_data.aisle_inventory[(s,a)],
                name=f"inventory_multi_{s}_{a}"
            )   

# _add_aisle_visit_linking_constraints
for i in model_data.container_ids:
    fixed_aisles = model_data.container_fixed_aisles.get(i, set())
    for a in fixed_aisles:
        for k in model_data.tour_indices:
            model.addConstr(v[a, k] >= x[i, k],name=f"fixed_aisle_{i}_{a}_{k}")

# Regular constraints for multi-location SKUs
for i in model_data.container_ids:
    for s in model_data.multi_location_skus:
        if (i,s) in model_data.container_sku_qty:
            for a in model_data.multi_location_skus[s]:
                for k in model_data.tour_indices:
                    model.addConstr(y[i,s,a,k] <= model_data.container_sku_qty[i,s] * z[i,s,a,k],name=f"linkyz_{i}_{s}_{a}_{k}")
                    model.addConstr(v[a, k] >= z[i,s,a,k],name=f"linkvz_{i}_{s}_{a}_{k}")

# _add_min_max_aisle_constraints
min_possible_aisle = min(min(model_data.sku_aisles[s]) for s in model_data.skus if model_data.sku_aisles[s])
max_possible_aisle = max(max(model_data.sku_aisles[s]) for s in model_data.skus if model_data.sku_aisles[s])

# Precompute potential min/max aisles for each container based on fixed aisles
for i in model_data.container_ids:
    fixed_aisles = model_data.container_fixed_aisles.get(i, set())
    if fixed_aisles:
        min_fixed = min(fixed_aisles)
        max_fixed = max(fixed_aisles)
        
        for k in model_data.tour_indices:
            # If container i is assigned to tour k, enforce its fixed aisle boundaries
            model.addGenConstrIndicator(x[i, k], 1,min_aisle_var[k] <= min_fixed,name=f"fixed_min_aisle_{i}_{k}")
            model.addGenConstrIndicator(x[i, k], 1,max_aisle_var[k] >= max_fixed,name=f"fixed_max_aisle_{i}_{k}")

# Standard constraints for all aisles visited (via v[a,k])
for k in model_data.tour_indices:
    for a in range(min_possible_aisle, max_possible_aisle + 1):
        model.addGenConstrIndicator(v[a, k], 1,min_aisle_var[k] <= a,name=f"indminaisle_v_{a}_{k}")
        model.addGenConstrIndicator(v[a, k], 1,max_aisle_var[k] >= a,name=f"indmaxaisle_v_{a}_{k}")

    # Ensure max_aisle is greater than min_aisle for active tours
    model.addConstr(max_aisle_var[k] >= min_aisle_var[k],name=f"aisle_order_{k}")

#TODO: Changes here
# _add_critical_container_constraints
critical_containers = getattr(model_data, 'critical_containers', set())
if critical_containers:
    # Only add constraints if we have critical containers
    critical_in_scope = set(model_data.container_ids).intersection(critical_containers)    
    if critical_in_scope:     
        for i in critical_in_scope:
            model.addConstr(gp.quicksum(x[i,k] for k in model_data.tour_indices) == 1,name=f"critical_{i}")

# _add_tour_ordering_constraints
for k in range(1, max(model_data.tour_indices) + 1):
    if k-1 in model_data.tour_indices and k in model_data.tour_indices:
        model.addConstr(u[k] <= u[k-1],name=f"tour_order_{k}")

# _add_valid_inequalities
        
# 1. Minimum number of aisles per tour based on containers
for k in model_data.tour_indices:
    model.addConstr(
        gp.quicksum(v[a, k] for a in range(min_possible_aisle, max_possible_aisle + 1)) >=  u[k], name=f"min_aisles_{k}")

# 2. Aggregated container-aisle relationship
for a in range(min_possible_aisle, max_possible_aisle + 1):
    containers_using_aisle = []
    for i in model_data.container_ids:
        if any(a in model_data.sku_aisles.get(s, []) for s in 
            [s for s in model_data.skus if (i,s) in model_data.container_sku_qty]):
            containers_using_aisle.append(i)
    
    if containers_using_aisle:
        for k in model_data.tour_indices:
            model.addConstr(v[a, k] <= gp.quicksum(x[i, k] for i in containers_using_aisle), name=f"aisle_container_link_{a}_{k}")

# 3. Upper bound on distinct aisles per tour
for k in model_data.tour_indices:
    unique_aisles_per_container = {}
    for i in model_data.container_ids:
        aisles = set()
        for s in model_data.skus:
            if (i,s) in model_data.container_sku_qty:
                aisles.update(model_data.sku_aisles.get(s, []))
        unique_aisles_per_container[i] = len(aisles)
    
    model.addConstr(
        gp.quicksum(v[a,k] for a in range(min_possible_aisle, max_possible_aisle + 1)) <= 
        gp.quicksum(unique_aisles_per_container[i] * x[i,k] for i in model_data.container_ids),
        name=f"tour_max_aisles_{k}"
    )
        
# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

# Get slack weights for containers
slack_weights = getattr(model_data, 'slack_weights', {})
    
# 1. Slack component (α) with slack weights
slack = gp.quicksum(slack_weights.get(i, 0.0) * x[i,k] for i in model_data.container_ids for k in model_data.tour_indices)
        
# 2.1 Distinct aisles component - count of distinct aisles visited per tour
distinct_aisles = gp.quicksum(v[a,k] for a, k in v.keys())
        
# 2.2 Aisle span component - max aisle minus min aisle
aisle_span = gp.quicksum((max_aisle_var[k] - min_aisle_var[k]) for k in model_data.tour_indices)
        
# 2.3 Combined travel distance
travel_distance = CONFIG['travel_distance_weight'] * (distinct_aisles + aisle_span)

# 2.4 Num_tours
num_tours = CONFIG['new_tour_weight']*gp.quicksum(u[k] for k in model_data.tour_indices)

#TODO: Changes here
# Set complete objective
model.setObjective(travel_distance - slack)#+ num_tours)# #num_tours, GRB.MINIMIZE)
        
model.update()

# ============================================================================
# SOLVE MODEL
# ============================================================================

# Solve
model.optimize()

# ============================================================================
# EXTRACT SOLUTION
# ============================================================================

# Check solution status
if model.SolCount > 0:
    print(f"Solution found with objective: {model.ObjVal}")
    
    # Extract solution
    solution = {
        'objective': model.ObjVal,
        'tours': {},
        'assignments': {}
    }
    
    # Extract tour assignments
    for i in model_data.container_ids:
        for k in model_data.tour_indices:
            if x[i,k].X > 0.5:  # Binary variable
                solution['assignments'][i] = k
                if k not in solution['tours']:
                    solution['tours'][k] = []
                solution['tours'][k].append(i)
    
    # Print results
    print("\nTour Formation Solution:")
    print(f"Objective Value: {solution['objective']}")
    #TODO: Changes here
    # Breakdown of objective value
    slack_term = sum(slack_weights.get(i, 0.0) * x[i,k].X for i in model_data.container_ids for k in model_data.tour_indices)
    travel_term = CONFIG['travel_distance_weight'] * sum(v[a,k].X for a, k in v.keys()) + sum((max_aisle_var[k].X - min_aisle_var[k].X) for k in model_data.tour_indices)
    tour_term = CONFIG['new_tour_weight'] * sum(u[k].X for k in model_data.tour_indices)
    print(f" Slack term: {-slack_term}")
    print(f" Travel term: {travel_term}")
    #print(f" Tour term: {tour_term}")
    print("\nTour Assignments:")
    for tour_id, containers in solution['tours'].items():
        volume_used = sum(model_data.container_volumes[c] for c in containers)
        min_aisle = int(min_aisle_var[tour_id].X)
        max_aisle = int(max_aisle_var[tour_id].X)
        distinct_aisles = sum(int(v[a, tour_id].X) for a in range(min_possible_aisle, max_possible_aisle + 1))
        travel_distance = CONFIG['travel_distance_weight'] * distinct_aisles + (max_aisle - min_aisle)
        print(f"Tour {tour_id}: Unutilized tour capacity {(CONFIG['max_volume_per_tour']-volume_used)*100/CONFIG['max_volume_per_tour']}%, Travel distance {travel_distance}, Containers {containers}")
    
    #Sanity check for unassigned containers
    assigned_containers = set(solution['assignments'].keys())
    unassigned_containers = [i for i in model_data.container_ids if i not in assigned_containers]
    print("\nUnassigned Containers:")
    print(unassigned_containers)
   
    # Print tours by highest median slack term first
    print("\nTour Pickup Locations (Aisles) (sorted by median slack):")
   
    # Compute median slack for each tour
    tour_slacks = {}
    for tour_id, containers in solution['tours'].items():
        slacks = [slack_weights.get(c, 0.0) for c in containers]
        if slacks:
            median_slack = np.median(slacks)
        else:
            median_slack = 0.0
        tour_slacks[tour_id] = median_slack

    # Sort tours by median slack descending
    sorted_tours = sorted(solution['tours'].items(), key=lambda x: tour_slacks[x[0]], reverse=True)

    for tour_id, containers in sorted_tours:
        aisle_sequence = []
        for a in range(min_possible_aisle, max_possible_aisle + 1):
            visit_count = int(round(v[a, tour_id].X))
            aisle_sequence.extend([a] * visit_count)
        if aisle_sequence:
            aisle_str = " -> ".join(f"aisle {a}" for a in aisle_sequence)
            print(f"Tour {tour_id} (median slack {tour_slacks[tour_id]}): {aisle_str}")
        else:
            print(f"Tour {tour_id} (median slack {tour_slacks[tour_id]}): Pickup aisles None")
    print('\ndone!')
else:
    print(f"No solution found. Status: {model.status}")