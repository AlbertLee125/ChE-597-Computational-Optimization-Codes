import pyomo.environ as pyo
import numpy as np

# Model setup
model = pyo.ConcreteModel()

# Time steps
K = 4  # k = 0, 1, 2, 3
time_steps = range(K)

# Parameters
A = np.array([[0.2681, -0.00338, -0.00728], [9.703, 0.3279, -25.44], [0, 0, 1]])
B = np.array([[-0.00537, 0.1655], [1.297, 97.91], [0, -6.637]])
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x_initial = np.array([-0.03, 0, 0.3])
csp, hsp = 0, 0  # Setpoints

# Variables
model.x = pyo.Var(time_steps, range(3), bounds=(-0.05, 0.05))  # State variables
model.u = pyo.Var(time_steps, range(2), bounds=(-10, 10))  # Input variables

# Constraints
def state_space_constraints(model, k, i):
    if k == 0:  # Initial condition
        return model.x[k, i] == x_initial[i]
    else:
        return model.x[k, i] == sum(A[i, j] * model.x[k-1, j] for j in range(3)) + sum(B[i, j] * model.u[k-1, j] for j in range(2))

model.dynamics = pyo.Constraint(time_steps, range(3), rule=state_space_constraints)

# Objective function
def objective_function(model):
    return sum(abs(C[0, i] * model.x[k, i] - csp) + abs(C[2, i] * model.x[k, i] - hsp) for k in time_steps for i in range(3))

model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Solver
solver = pyo.SolverFactory('ipopt')
results = solver.solve(model)

# Display results
for k in time_steps:
    for i in range(3):
        print(f'x[{k},{i}] = {model.x[k,i].value}')
for k in time_steps[:-1]:  # No input at the last time step
    for i in range(2):
        print(f'u[{k},{i}] = {model.u[k,i].value}')
print(f'Objective function value = {model.objective.values}')  # Objective function value