import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plot_feasible_region_with_path(path):
    # Intersection points
    intersection_points = [(1/2 - np.sqrt(3)/2, -np.sqrt(3)/2 - 1/2), (1/2 + np.sqrt(3)/2, np.sqrt(3)/2 - 1/2)]

    # x1 and x2 non-negative real
    x1 = np.linspace(0, 2, 100)
    x2 = np.linspace(0, 2, 100)

    # Create meshgrid
    X1, X2 = np.meshgrid(x1, x2)

    # Constraints x1^2 + x2^2 <= 2 and x1 - x2 <= 1
    F = X1**2 + X2**2 - 2
    G = X1 - X2 - 1

    # Plot the constraints
    plt.figure(figsize=(6, 6))
    plt.contour(X1, X2, F, [0], colors='r')
    plt.contour(X1, X2, G, [0], colors='b')

    # Correctly fill the feasible region
    x_fill = np.linspace(intersection_points[0][0], intersection_points[1][0], 100)
    y_fill_lower = np.maximum(0, x_fill - 1)
    y_fill_upper = np.sqrt(2 - x_fill**2)
    plt.fill_between(x_fill, y_fill_lower, y_fill_upper, color='grey', alpha=0.5)

    # Plot the path with arrows
    for i in range(len(path) - 1):
        plt.annotate('', xy=path[i + 1], xytext=path[i], arrowprops=dict(arrowstyle='->', color='black'))

    # Annotate the final point
    plt.scatter(*path[-1], color='black')
    plt.text(path[-1][0], path[-1][1], f'Final: ({path[-1][0]:.4f}, {path[-1][1]:.4f})', fontsize=9)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.title('Feasible Region with Optimization Path')
    plt.grid(True)
    plt.show()

# Optimal solution: x1 = 0.6179, x2 = 0.8317
# Optimal objective: 4.8020

# def primal_dual_interior_point_v1(x0, t0=1, mu=1.2, epsilon=1e-5, alpha=0.5, beta=0.3):
#     def objective(x):
#         x1, x2 = x
#         return 3 / (x1 + x2) + x2 + np.exp(x1) + (x1 - x2)**2

#     def gradient_objective(x):
#         x1, x2 = x
#         return np.array([
#             -3 / (x1 + x2)**2 + np.exp(x1) + 2 * (x1 - x2),
#             3 / (x1 + x2)**2 + 1 - 2 * (x1 - x2)
#         ])

#     def constraints(x):
#         return np.array([
#             -x[0],  # x1 >= 0
#             -x[1],  # x2 >= 0
#             x[0]**2 + x[1]**2 - 2,  # x1^2 + x2^2 <= 2
#             x[0] - x[1] - 1  # x1 - x2 <= 1
#         ])

#     def jacobian_constraints(x):
#         return np.array([
#             [-1, 0],  # Derivative of -x1
#             [0, -1],  # Derivative of -x2
#             [2 * x[0], 2 * x[1]],  # Derivative of x1^2 + x2^2 - 2
#             [1, -1]  # Derivative of x1 - x2 - 1
#         ])

#     x_current = np.array(x0)
#     lambda_current = np.zeros(len(constraints(x0)))  # Initialize lambda for the number of constraints
#     path = [x_current.tolist()]  # Track the path of x

#     while True:
#         grad_L = gradient_objective(x_current) + np.dot(jacobian_constraints(x_current).T, lambda_current)
#         H = np.eye(len(x0))  # Approximate Hessian with identity matrix for simplicity

#         kkt_matrix = np.block([
#             [H, jacobian_constraints(x_current).T],
#             [jacobian_constraints(x_current), np.zeros((len(lambda_current), len(lambda_current)))]
#         ])

#         kkt_rhs = -np.hstack([grad_L, -constraints(x_current)])

#         try:
#             delta = np.linalg.solve(kkt_matrix, kkt_rhs)
#         except np.linalg.LinAlgError:
#             print("Singular KKT matrix encountered. Adjusting the approach might be necessary.")
#             break

#         delta_x = delta[:len(x0)]
#         delta_lambda = delta[len(x0):]

#         x_current += delta_x
#         lambda_current += delta_lambda
#         path.append(x_current.tolist())

#         if np.linalg.norm(delta_x) < epsilon and np.linalg.norm(delta_lambda) < epsilon:
#             break

#     return x_current, path

# # Initial guess
# x0 = [0.5, 0.5]

# # Solve the optimization problem using primal-dual interior point method version 1
# x_opt, path = primal_dual_interior_point_v1(x0)

# print("Optimal Solution:", x_opt)
# print("Path:", path)

import numpy as np

def barrier_function(x, t):
    """Barrier-augmented objective function."""
    obj = 3 / (x[0] + x[1]) + np.exp(x[0]) + (x[0] - x[1]) ** 2
    barrier = -np.log(x[0]) - np.log(x[1]) - np.log(2 - x[0] ** 2 - x[1] ** 2) - np.log(1 - (x[0] - x[1]))
    return t * obj - barrier

def gradient(x, t):
    """Gradient of the barrier-augmented objective function."""
    grad = np.array([
        t * (-3 / (x[0] + x[1]) ** 2 + np.exp(x[0]) + 2 * (x[0] - x[1])) - (1 / x[0]) - (2 * x[0]) / (2 - x[0] ** 2 - x[1] ** 2) - (1 / (1 - (x[0] - x[1]))),
        t * (-3 / (x[0] + x[1]) ** 2 - 2 * (x[0] - x[1])) - (1 / x[1]) - (2 * x[1]) / (2 - x[0] ** 2 - x[1] ** 2) + (1 / (1 - (x[0] - x[1])))
    ])
    return grad

def hessian(x, t):
    """Hessian of the barrier-augmented objective function."""
    H = np.array([
        [t * (6 / (x[0] + x[1]) ** 3 + np.exp(x[0]) + 2) + (1 / x[0] ** 2) + (2 * (2 - x[1] ** 2)) / (2 - x[0] ** 2 - x[1] ** 2) ** 2 + (1 / (1 - (x[0] - x[1])) ** 2), t * (6 / (x[0] + x[1]) ** 3 - 2) - (2 * x[0] * x[1]) / (2 - x[0] ** 2 - x[1] ** 2) ** 2 - (1 / (1 - (x[0] - x[1])) ** 2)],
        [t * (6 / (x[0] + x[1]) ** 3 - 2) - (2 * x[0] * x[1]) / (2 - x[0] ** 2 - x[1] ** 2) ** 2 - (1 / (1 - (x[0] - x[1])) ** 2), t * (6 / (x[0] + x[1]) ** 3 + 2) + (1 / x[1] ** 2) + (2 * (2 - x[0] ** 2)) / (2 - x[0] ** 2 - x[1] ** 2) ** 2 + (1 / (1 - (x[0] - x[1])) ** 2)]
    ])
    return H

def backtracking_line_search(x, t, f, grad_f, dx, alpha, beta):
    """Backtracking line search to find the step size."""
    step = 1
    while f(x + step * dx, t) > f(x, t) + alpha * step * np.dot(grad_f(x, t).T, dx):
        step *= beta
    return step

# Initial parameters
x = np.array([0.5, 0.5])  # Initial guess for x
t = 1.0  # Initial t
mu = 1.2  # Update factor for t
epsilon = 1e-5  # Convergence threshold
alpha = 0.5  # Backtracking line search parameter
beta = 0.3  # Backtracking line search parameter

# For tracking the path to convergence
path = [x.copy()]

# Main optimization loop
while np.linalg.norm(gradient(x, t)) > epsilon:
    # Compute the search direction using Newton's method
    dx = -np.linalg.inv(hessian(x, t)).dot(gradient(x, t))
    
    # Backtracking line search to find the step size
    step = backtracking_line_search(x, t, barrier_function, gradient, dx, alpha, beta)
    
    # Update the primal variables
    x += step * dx
    path.append(x.copy())
    
    # Update t
    t *= mu

    # Check for convergence
    if step < epsilon:
        break

# Final solution and the path taken
x, path
print("Optimal Solution:", x)
print("Path:", path)
