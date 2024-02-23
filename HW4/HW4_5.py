import numpy as np
from numpy.linalg import eigvals

# Define the matrix Q
Q = np.array([[0, 1, 3, 1], [1, 0, 0, 2], [3, 0, 0, 4], [1, 2, 4, 0]])

# Initialize variables to keep track of the best solution
max_value = -np.inf
best_X = None

# Generate symmetric matrices with ones on the diagonal
for _ in range(10000):  # Iterate over a number of random matrices
    # Generate a random symmetric matrix
    A = np.random.randn(4, 4)
    A_sym = (A + A.T) / 2
    
    # Set the diagonal to 1s
    np.fill_diagonal(A_sym, 1)

    # Check if the matrix is positive semidefinite (all eigenvalues are non-negative)
    if np.all(eigvals(A_sym) >= 0):
        # Compute the objective function value
        value = np.trace(Q @ A_sym)
        
        # Update the best solution if this is the highest value found so far
        if value > max_value:
            max_value = value
            best_X = A_sym

# Print the best solution found and max_value rounded by 3 decimal places
print("Best solution found:")
print(best_X.round(3))
print("Objective function value:", max_value.round(3))


# import numpy as np
# from gurobipy import Model, GRB, quicksum

# # Define the Q matrix as given in the problem
# Q = np.array([[0, 1, 3, 1], [1, 0, 0, 0], [3, 0, 0, 2], [1, 0, 2, 4]])

# # Function to check the positive semidefiniteness of a matrix
# def is_positive_semidefinite(X, tolerance=-1e-4):
#     eigenvalues = np.linalg.eigvalsh(X)
#     min_eigenvalue = np.min(eigenvalues)
#     return min_eigenvalue > tolerance

# # Initialize the model
# model = Model()

# # Define the size of the matrix X
# n = 4

# # Add variables for X
# X_vars = model.addVars(n, n, lb=-GRB.INFINITY, name="X")

# # Set the objective function to maximize <Q, X>
# model.setObjective(quicksum(Q[i, j] * X_vars[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

# # Add constraints for the diagonal elements of X to be 1
# for i in range(n):
#     model.addConstr(X_vars[i, i] == 1, f"diag_{i}")

# # Placeholder for iterative addition of cutting planes based on eigenvalue decomposition

# # Optimize the model
# model.optimize()

# # Check if the optimization was successful
# if model.Status == GRB.OPTIMAL:
#     # Extract the solution matrix X
#     solution_X = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             solution_X[i, j] = X_vars[i, j].X

#     # Check if the solution X is positive semidefinite
#     if is_positive_semidefinite(solution_X):
#         print("Solution is positive semidefinite.")
#         print("Solution X:", solution_X)
#     else:
#         print("Solution X is not positive semidefinite. Further iterations needed.")
# else:
#     print("Optimization was not successful. Status code:", model.Status)
