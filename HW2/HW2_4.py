import pyomo.environ as pyo

# Create a concrete model
model = pyo.ConcreteModel()

# Sets for nodes, edges, and commodities
nodes = {'s1', 's2', 's3', 't1', 't2', 't3', 'x', 'y'}
edges = {('s1', 'x'), ('s1', 't1'), ('s2', 'x'), ('s3', 'x'), ('s3', 't3'), ('x', 'y'), ('y', 't1'), ('y', 't2'), ('y', 't3')}
commodities = {'d1', 'd2', 'd3'}

# Parameters: capacities, costs, demands, and supply
capacities = {('s1', 'x'):5, ('s1', 't1'):7, ('s2', 'x'):3, ('s3', 'x'):5, ('s3', 't3'):5, ('x', 'y'):14, ('y', 't1'):3, ('y', 't2'):3, ('y', 't3'):4}
costs = {('s1', 'x'):1, ('s1', 't1'):3, ('s2', 'x'):2, ('s3', 'x'):2, ('s3', 't3'):5, ('x', 'y'):2, ('y', 't1'):1, ('y', 't2'):2, ('y', 't3'):2}
demands = {'d1': 9, 'd2': 2, 'd3': 3}

# Variables
model.flow = pyo.Var(commodities, edges, within=pyo.NonNegativeReals)

# Objective function
def objective_rule(model):
    return sum(costs[e] * model.flow[k, e] for k in commodities for e in edges)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints
# Capacity constraints
def capacity_rule(model, i, j):
    return sum(model.flow[k, (i, j)] for k in commodities) <= capacities[(i, j)]

model.capacity_constraints = pyo.Constraint(edges, rule=capacity_rule)

# Flow conservation constraints
# Map commodities to their source and sink nodes
source = {'d1': 's1', 'd2': 's2', 'd3': 's3'}
sink = {'d1': 't1', 'd2': 't2', 'd3': 't3'}

def flow_conservation_rule(model, n, k):
    if n == source[k]:  # Source node for commodity k
        return sum(model.flow[k, (i, j)] for i, j in edges if i == n) == demands[k]
    elif n == sink[k]:  # Sink node for commodity k
        return sum(model.flow[k, (i, j)] for i, j in edges if j == n) - sum(model.flow[k, (i, j)] for i, j in edges if i == n) == -demands[k]
    else:  # Intermediate nodes
        return sum(model.flow[k, (i, j)] for i, j in edges if j == n) - sum(model.flow[k, (i, j)] for i, j in edges if i == n) == 0

model.flow_conservation = pyo.Constraint(nodes, commodities, rule=flow_conservation_rule)

# Solve
solver = pyo.SolverFactory('gurobi')  
solution = solver.solve(model, tee=True)

# Check if the solution is feasible
if (solution.solver.status == pyo.SolverStatus.ok) and (solution.solver.termination_condition == pyo.TerminationCondition.optimal):
    # Print the results
    for k in commodities:
        for e in edges:
            print(f"Flow of commodity {k} through edge {e}: {model.flow[k, e].value}")
else:
    print("No feasible solution found")


