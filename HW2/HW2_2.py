# # This is the the pyomo model solved with IPOPT to verify whether the solution is optimal.
# # You can uncomment the following code to run it.
# import pyomo.environ as pyo
# from pyomo.opt import SolverFactory

# # Define the function f
# def f(model):
#     x1 = model.x1
#     x2 = model.x2
#     x3 = model.x3
#     return x3 * pyo.log(pyo.exp(x1/x3) + pyo.exp(x2/x3)) + (x3 - 2)**2 + pyo.exp(1/(x1 + x2))

# # Create a model instance
# model = pyo.ConcreteModel()

# # Define variables
# model.x1 = pyo.Var(initialize=1, within=pyo.Reals)
# model.x2 = pyo.Var(initialize=1, within=pyo.Reals)
# model.x3 = pyo.Var(initialize=1, within=pyo.PositiveReals)

# # Add objective
# model.objective = pyo.Objective(rule=f, sense=pyo.minimize)

# # Add constraints
# model.constraint1 = pyo.Constraint(expr=model.x1 + model.x2 >= 0)
# model.constraint2 = pyo.Constraint(expr=model.x3 >= 0)

# # Specify the IPOPT solver
# opt = SolverFactory('ipopt')

# # Solve the model
# results = opt.solve(model, tee=True)

# # Get the results
# x1_opt = pyo.value(model.x1)
# x2_opt = pyo.value(model.x2)
# x3_opt = pyo.value(model.x3)
# min_f = pyo.value(model.objective)

# print('x1 =', x1_opt)
# print('x2 =', x2_opt)
# print('x3 =', x3_opt)
# print('min_f =', min_f)

# Problem 2 Quasi-Newton method (BFGS method)
# You can uncomment the following code to run it.
import numpy as np

# Define the function f
def f(x1, x2, x3):
    return x3 * np.log(np.exp(x1/x3) + np.exp(x2/x3)) + (x3 - 2)**2 + np.exp(1/(x1 + x2))

# Gradient of the function f for numerical approximation
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros(3)
    for i in range(3):
        x_plus_h = np.copy(x)
        x_plus_h[i] += h
        grad[i] = (f(*x_plus_h) - f(*x)) / h
    return grad

# Hessian of the function f for numerical approximation
def numerical_hessian(f, x, h=1e-5):
    hessian = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            x_plus_h = np.copy(x)
            x_plus_h[i] += h
            x_plus_h[j] += h
            hessian[i, j] = (f(*x_plus_h) - f(*x) - numerical_gradient(f, x, h)[i]*h - numerical_gradient(f, x, h)[j]*h) / (h**2)
    return hessian

# Quasi-Newton method (BFGS method)
def quasi_newton_method(f, x_init, h=1e-5, tol=1e-5, max_iter=10000):
    x = np.copy(x_init)
    grad = numerical_gradient(f, x, h)
    H = np.linalg.inv(numerical_hessian(f, x, h))
    for i in range(max_iter):
        p = -H @ grad
        x_new = x + p
        if np.linalg.norm(x_new - x) < tol:
            break
        grad_new = numerical_gradient(f, x_new, h)
        s = x_new - x
        y = grad_new - grad
        rho = 1.0 / (y @ s)
        H = (np.eye(3) - rho * np.outer(s, y)) @ H @ (np.eye(3) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        grad = grad_new
    return x

# Run quasi-Newton method (BFGS method)
x_init = np.array([1.1, 1.1, 1.1])
x1, x2, x3 = quasi_newton_method(f, x_init)
print('x1 =', x1)
print('x2 =', x2)
print('x3 =', x3)
print('min_f =', f(x1, x2, x3))
