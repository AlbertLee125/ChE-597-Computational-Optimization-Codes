import pyomo.environ as pyo

# Data
nodes = [1, 2, 3, 4, 5, 6, 7, 8]
edges = [
    (1, 8),
    (1, 3),
    (1, 4),
    (2, 8),
    (2, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (4, 3),
    (4, 5),
    (4, 6),
    (4, 7),
]
demands = {1: -1000, 2: -1000, 3: 0, 4: 0, 5: 450, 6: 500, 7: 610, 8: 440}
costs = {
    (1, 8): 0,
    (1, 3): 7,
    (1, 4): 8,
    (2, 8): 0,
    (2, 3): 4,
    (2, 4): 7,
    (3, 4): 0,
    (3, 5): 25,
    (3, 6): 6,
    (3, 7): 17,
    (4, 3): 0,
    (4, 5): 29,
    (4, 6): 8,
    (4, 7): 5,
}
capacities = {
    (1, 8): float('inf'),
    (1, 3): float('inf'),
    (1, 4): float('inf'),
    (2, 8): float('inf'),
    (2, 3): float('inf'),
    (2, 4): float('inf'),
    (3, 4): 25,
    (3, 5): float('inf'),
    (3, 6): float('inf'),
    (3, 7): float('inf'),
    (4, 3): 25,
    (4, 5): float('inf'),
    (4, 6): float('inf'),
    (4, 7): float('inf'),
}

# Model
model = pyo.ConcreteModel()

# Sets
model.Nodes = pyo.Set(initialize=nodes)
model.Edges = pyo.Set(dimen=2, initialize=edges)

# Parameters
model.Demands = pyo.Param(model.Nodes, initialize=demands)
model.Costs = pyo.Param(model.Edges, initialize=costs)
model.Capacities = pyo.Param(model.Edges, initialize=capacities)

# Variables
model.Flow = pyo.Var(model.Edges, within=pyo.NonNegativeReals)


# Objective
def objective_function(model):
    return sum(model.Costs[e] * model.Flow[e] for e in model.Edges)


model.Objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)


# Constraints
def flow_conservation_rule(model, node):
    inflows = sum(model.Flow[(i, node)] for i in nodes if (i, node) in edges)
    outflows = sum(model.Flow[(node, j)] for j in nodes if (node, j) in edges)
    return inflows - outflows == model.Demands[node]


model.FlowConservation = pyo.Constraint(model.Nodes, rule=flow_conservation_rule)


def capacity_constraints(model, i, j):
    return model.Flow[(i, j)] <= model.Capacities[(i, j)]


model.CapacityConstraints = pyo.Constraint(model.Edges, rule=capacity_constraints)

# Solve using Gurobi with the primal simplex algorithm
solver = pyo.SolverFactory('gurobi')
solver.options['Method'] = 0  # For primal simplex
results = solver.solve(model, tee=True)

# Display results
model.display()
print(results)

# Display the flow on each edge
for edge in edges:
    print(f"Flow on edge {edge}: {model.Flow[edge].value}")

# Display the total cost
print(f"Total cost: {model.Objective():,.2f}")
