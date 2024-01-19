import pyomo.environ as pyo
import numpy as np

def model():
    # Given costs for each node
    costs = {1: 40, 2: 65, 3: 43, 4: 48, 5: 72, 6: 36}

    # Adjacency matrix representing the connections
    adj_matrix = np.array(
        [
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0],
        ]
    )

    # Create the model
    m = pyo.ConcreteModel()

    # Define the binary variables for placing monitoring stations
    m.x = pyo.Var(range(1, 7), domain=pyo.Binary)

    # Define the objective function to minimize the total cost
    def obj(m):
        return sum(costs[i] * m.x[i] for i in range(1, 7))
    
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # Define the constraints based on the adjacency matrix
    m.coverage = pyo.ConstraintList()

    for i in range(6):
        for j in range(i, 6):
            if adj_matrix[i][j] == 1:
                m.coverage.add(m.x[i + 1] + m.x[j + 1] >= 1) # Python is 0-indexed

    return m

if __name__ == '__main__':
    m = model()
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(m)
    costs = {1: 40, 2: 65, 3: 43, 4: 48, 5: 72, 6: 36}
    # Instead of m.display(), print the results
    print("The optimal station placement is:")
    for i in range(1, 7):
        print(f"Station at node {i}: {'Yes' if pyo.value(m.x[i]) > 0.5 else 'No'}")

    print("\n The minimum total cost is:")
    print(sum(costs[i] * pyo.value(m.x[i]) for i in range(1, 7)))
