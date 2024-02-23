import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def barrier_method_bfgs(x0, t0=1, mu=4, epsilon=1e-4):
    def objective(x):
        x1, x2 = x
        return 3 / (x1 + x2) + x2 + np.exp(x1) + (x1 - x2)**2

    def barrier(x, t):
        x1, x2 = x
        # Using very small values to avoid log(0)
        return -1/t * (np.log(max(x1, 1e-10)) + np.log(max(x2, 1e-10)) + 
                       np.log(max(2 - x1**2 - x2**2, 1e-10)) + 
                       np.log(max(1 - x1 + x2, 1e-10)))

    def augmented_objective(x, t):
        return objective(x) + barrier(x, t)
    
    path = [x0]  # To track the path of x
    t = t0
    x_current = np.array(x0)

    while 1/t > epsilon:
        # Minimize the augmented objective function using the BFGS method
        res = minimize(lambda x: augmented_objective(x, t), x_current, method='BFGS', options={'disp': False})
        
        x_current = res.x
        path.append(x_current.tolist())  # Update the path
        
        t *= mu  # Update the barrier parameter
    
    return x_current, path

def objective(x):
        x1, x2 = x
        return 3 / (x1 + x2) + x2 + np.exp(x1) + (x1 - x2)**2

# Initial guess
x0 = [0.5, 0.5]

# Solve the optimization problem using the barrier method with BFGS
x_opt, path = barrier_method_bfgs(x0)

# Rounding the optimal solution to 4 decimal places
x_opt_rounded = [round(num, 4) for num in x_opt]

print("Optimal Solution:", x_opt_rounded)
print("Path:", [list(map(lambda x: round(x, 4), coords)) for coords in path])

# Assuming the objective value needs to be calculated and rounded at the optimal solution
optimal_objective = objective(x_opt)
print("Optimal Objective:", round(optimal_objective, 4))

# Optimal solution: x1 = 0.6179, x2 = 0.8317
# Optimal objective: 4.8020

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

# Given path from the optimization output
path = [[0.5, 0.5], [0.5732, 0.8584], [0.5997, 0.8428], [0.6125, 0.8351], [0.6165, 0.8326], [0.6176, 0.8319], [0.6178, 0.8318], [0.6179, 0.8317]]

# Call the function with the optimization path
plot_feasible_region_with_path(path)
# The feasible region is plotted with the intersection points and the path of the optimization algorithm. The path is plotted with arrows and the final point is annotated.