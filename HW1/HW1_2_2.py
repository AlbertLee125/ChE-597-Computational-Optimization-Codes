import pyomo.environ as pyo
import numpy as np

def create_model():
    # Adjacency matrix representing the connections
    adj_matrix = np.array([
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0]
    ])

    # Create the model
    m = pyo.ConcreteModel()

    # Define the binary variables for placing monitoring stations
    m.x = pyo.Var(range(1, 7), within=pyo.Binary)

    # Define binary variables for each road link to indicate if it is uncovered
    m.y = pyo.Var([(i+1, j+1) for i in range(6) for j in range(i+1, 6)], within=pyo.Binary)

    # Objective: Minimize the number of uncovered road links
    m.objective = pyo.Objective(expr=sum(m.y[i, j] for (i, j) in m.y), sense=pyo.minimize)

    # Constraint: At most 2 monitoring stations
    m.station_limit = pyo.Constraint(expr=sum(m.x[i] for i in range(1, 7)) <= 2)

    # Constraints for road link coverage
    m.coverage = pyo.ConstraintList()
    for i in range(6):
        for j in range(i+1, 6):
            if adj_matrix[i][j] == 1:
                m.coverage.add(expr=m.x[i+1] + m.x[j+1] + m.y[i+1, j+1] >= 1)

    return m

# Create and solve the model
m = create_model()
solver = pyo.SolverFactory('gurobi')
result = solver.solve(m)

# Print the results
print("Optimal solution found:")
for i in range(1, 7):
    print(f"Node {i} station: {'Yes' if pyo.value(m.x[i]) > 0.5 else 'No'}")
uncovered_count = sum(pyo.value(m.y[i, j]) for (i, j) in m.y)
print(f"Number of uncovered road links: {uncovered_count}")