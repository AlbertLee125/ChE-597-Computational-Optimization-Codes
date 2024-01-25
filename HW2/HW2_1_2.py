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

# Problem 1.2 part (i) Gradient descent with backtracking line search
# You can uncomment the following code to run it.
import numpy as np

# Define the function f
def f(x1, x2, x3):
    return x3 * np.log(np.exp(x1/x3) + np.exp(x2/x3)) + (x3 - 2)**2 + np.exp(1/(x1 + x2))

# Gradient of the function f for numerical approximation
def numerical_gradient(f, x1, x2, x3, h=1e-5):
    grad = np.zeros(3)
    for i in range(3):
        x = np.array([x1, x2, x3])
        x[i] += h
        grad[i] = (f(*x) - f(x1, x2, x3)) / h
    
    return grad

# Backtracking line search
def backtracking_line_search(f, x1, x2, x3, grad, alpha=0.4, beta=0.5):
    t = 1
    while f(x1 - t*grad[0], x2 - t*grad[1], x3 - t*grad[2]) > f(x1, x2, x3) - alpha*t*np.linalg.norm(grad)**2:
        t *= beta
    
    return t

# Gradient descent with backtracking line search
def gradient_descent(f, x1, x2, x3, alpha=0.4, beta=0.5, max_iter=1000, tol=1e-5):
    for i in range(max_iter):
        grad = numerical_gradient(f, x1, x2, x3)
        if np.linalg.norm(grad) < tol:
            break
        t = backtracking_line_search(f, x1, x2, x3, grad, alpha, beta)
        x1 -= t*grad[0]
        x2 -= t*grad[1]
        x3 -= t*grad[2]
    
    return x1, x2, x3

# Run gradient descent with backtracking line search
x1, x2, x3 = gradient_descent(f, 1.1, 1.1, 1.1)
print('x1 =', x1)
print('x2 =', x2)
print('x3 =', x3)
print('min_f =', f(x1, x2, x3))

# Problem 1.2 part (ii) Newton's method
# You can uncomment the following code to run it.
import numpy as np

# Define the function f
def f(x1, x2, x3):
    return x3 * np.log(np.exp(x1/x3) + np.exp(x2/x3)) + (x3 - 2)**2 + np.exp(1/(x1 + x2))

# Gradient of the function f for numerical approximation
def numerical_gradient(f, x1, x2, x3, h=1e-5):
    grad = np.zeros(3)
    for i in range(3):
        x = np.array([x1, x2, x3])
        x[i] += h
        grad[i] = (f(*x) - f(x1, x2, x3)) / h
    
    return grad

# Hessian of the function f for numerical approximation
def numerical_hessian(f, x1, x2, x3, h=1e-5):
    hessian = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            x = np.array([x1, x2, x3])
            x[i] += h
            x[j] += h
            hessian[i, j] = (f(*x) - f(x1, x2, x3) - f(x1 + h, x2 - h, x3 - h) + f(x1 + h, x2, x3) + f(x1, x2 - h, x3) + f(x1, x2, x3 - h)) / h**2
    
    return hessian

# Newton's method
def newton(f, x1, x2, x3, max_iter=10000, tol=1e-5):
    for i in range(max_iter):
        grad = numerical_gradient(f, x1, x2, x3)
        hessian = numerical_hessian(f, x1, x2, x3)
        if np.linalg.norm(grad) < tol:
            break
        x = np.array([x1, x2, x3])
        x -= np.linalg.inv(hessian) @ grad
        x1, x2, x3 = x
    
    return x1, x2, x3

# Run Newton's method
x1, x2, x3 = newton(f, 1.1, 1.1, 1.1)
print('x1 =', x1)
print('x2 =', x2)
print('x3 =', x3)
print('min_f =', f(x1, x2, x3))
