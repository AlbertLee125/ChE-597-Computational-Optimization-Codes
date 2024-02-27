import numpy as np

def primal_dual_interior_point_v2(x0, u0, mu=1.2, epsilon=1e-5, alpha=0.5, beta=0.3):
    def objective(x):
        x1, x2 = x
        return 3 / (x1 + x2) + x2 + np.exp(x1) + (x1 - x2)**2

    def gradient_objective(x):
        x1, x2 = x
        return np.array([-3 / ((x1 + x2)**2)+ np.exp(x1) + 2 * (x1 - x2), -3 / ((x1 + x2)**2) - 2 * (x1 - x2)])

    def hessian_objective(x):
        x1, x2 = x
        return np.array([[6 / (x1 + x2)**3 + np.exp(x1) + 2, -6 / (x1 + x2)**3 - 2], [6 / (x1 + x2)**3 - 2, 6 / (x1 + x2)**3 + 2]])

    def constraints(x):
        return np.array([
            x[0],  # x1 >= 0
            x[1],  # x2 >= 0
            2 - (x[0]**2 + x[1]**2),  # x1^2 + x2^2 <= 2
            1 - (x[0] - x[1])  # x1 - x2 <= 1
        ])

    def jacobian_constraints(x):
        return np.array([
            [1, 0],  # Derivative w.r.t x1 of x1
            [0, 1],  # Derivative w.r.t x2 of x2
            [-2 * x[0], -2 * x[1]],  # Derivative of x1^2 + x2^2 - 2
            [-1, 1]  # Derivative of x1 - x2 - 1
        ])

    x_current = np.array(x0)
    u_current = np.array(u0)  # Dual variables
    path = [x_current.tolist()]

    while True:
        grad_L = gradient_objective(x_current) + np.dot(jacobian_constraints(x_current).T, u_current)
        hess_L = hessian_objective(x_current)
        
        A = jacobian_constraints(x_current)
        zero_matrix = np.zeros((4, 4))  # Adjusted to have 4 rows and 4 columns

        # Form the KKT matrix with corrected dimensions
        kkt_matrix = np.block([[hess_L, A.T],[A, zero_matrix]])
        kkt_rhs = -np.hstack([grad_L, -constraints(x_current)])
        delta = np.linalg.solve(kkt_matrix, kkt_rhs)

        # Update x and u using backtracking line search
        step_size = 1
        while True:
            new_x = x_current + step_size * delta[:2]
            new_u = u_current + step_size * delta[2:]
            new_primal_residual = constraints(new_x)
            new_dual_residual = gradient_objective(new_x) + np.dot(jacobian_constraints(new_x).T, new_u)
            if np.all(new_primal_residual >= 0) and np.linalg.norm(new_primal_residual) < np.linalg.norm(new_primal_residual) and np.linalg.norm(new_dual_residual) < np.linalg.norm(new_dual_residual):
                break
            step_size *= beta

        x_current = new_x
        u_current = new_u
        path.append(x_current.tolist())

    return x_current, path


# Initial guess
x0 = [0.5, 0.5]
u0 = [1, 1, 1, 1]  # Initial guess for the dual variables

# Solve the optimization problem using primal-dual interior point method version 2
x_opt, path = primal_dual_interior_point_v2(x0, u0)

print("Optimal Solution:", x_opt)
print("Path:", path)

