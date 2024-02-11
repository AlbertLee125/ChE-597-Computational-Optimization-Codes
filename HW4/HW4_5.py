import numpy as np
from gurobipy import Model, GRB, quicksum

# Define the Q matrix as given in the problem
Q = np.array([[0, 1, 3, 1], [1, 0, 0, 0], [3, 0, 0, 2], [1, 0, 2, 4]])

# Function to check the positive semidefiniteness of a matrix
def is_positive_semidefinite(X, tolerance=-1e-4):
    eigenvalues = np.linalg.eigvalsh(X)
    min_eigenvalue = np.min(eigenvalues)
    return min_eigenvalue > tolerance

# Initialize the model
model = Model()

# Define the size of the matrix X
n = 4

# Add variables for X
X_vars = model.addVars(n, n, lb=-GRB.INFINITY, name="X")

# Set the objective function to maximize <Q, X>
model.setObjective(quicksum(Q[i, j] * X_vars[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

# Add constraints for the diagonal elements of X to be 1
for i in range(n):
    model.addConstr(X_vars[i, i] == 1, f"diag_{i}")

# Placeholder for iterative addition of cutting planes based on eigenvalue decomposition

# Optimize the model
model.optimize()

# Check if the optimization was successful
if model.Status == GRB.OPTIMAL:
    # Extract the solution matrix X
    solution_X = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            solution_X[i, j] = X_vars[i, j].X

    # Check if the solution X is positive semidefinite
    if is_positive_semidefinite(solution_X):
        print("Solution is positive semidefinite.")
        print("Solution X:", solution_X)
    else:
        print("Solution X is not positive semidefinite. Further iterations needed.")
else:
    print("Optimization was not successful. Status code:", model.Status)
