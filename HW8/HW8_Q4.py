import numpy as np
import pyomo.environ as pyo

# Load the data
num_data_points = 20
num_dimensions = 10
data = np.loadtxt('data_HW8_Q4.csv', delimiter=',')
num_clusters = 3

# Create a Pyomo model
model = pyo.ConcreteModel()

# Indices for the data points and clusters
model.I = pyo.RangeSet(num_data_points)
model.J = pyo.RangeSet(num_clusters)

# Decision variables
# Binary variable for data point i assigned to cluster j
model.x = pyo.Var(model.I, model.J, within=pyo.Binary)
# Continuous variable for the coordinates of cluster centroids
model.c = pyo.Var(model.J, pyo.RangeSet(num_dimensions), within=pyo.Reals)

# Objective: Minimize the sum of squared distances from points to their cluster centroids
def objective_rule(model):
    return sum(model.x[i,j] * sum((data[i-1,k] - model.c[j,k+1])**2 for k in range(num_dimensions)) for i in model.I for j in model.J)
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraint: Each data point must be assigned to exactly one cluster
def one_cluster_rule(model, i):
    return sum(model.x[i,j] for j in model.J) == 1
model.one_cluster_constraint = pyo.Constraint(model.I, rule=one_cluster_rule)

solver = pyo.SolverFactory('ipopt')  # Using Gurobi solver
result = solver.solve(model)

# Output the results
# Display the cluster assignments
for i in model.I:
    for j in model.J:
        if pyo.value(model.x[i,j]) > 0.5:  # Assuming x[i,j] is binary, so >0.5 means it's assigned
            print(f"Data point {i} is assigned to cluster {j}")

# Display the centroid coordinates
for j in model.J:
    centroid = [pyo.value(model.c[j,k+1]) for k in range(num_dimensions)]
    print(f"Centroid of cluster {j}: {centroid}")
