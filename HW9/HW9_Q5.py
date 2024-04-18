import pyomo.environ as pyo

m = pyo.ConcreteModel()

# Define variables with correct bounds within the model
m.x1 = pyo.Var(bounds=(0, 4), domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(bounds=(-1, 2), domain=pyo.Reals)
m.x3 = pyo.Var(bounds=(0, 3), domain=pyo.NonNegativeReals)
m.x4 = pyo.Var(bounds=(-1, 1), domain=pyo.Reals)

# Define the first constraint using model's variables
def constraint1(m):
    return 2 * m.x1 + m.x2 - m.x3 - 2 * m.x4 == 1

m.c1 = pyo.Constraint(rule=constraint1)

# Correct the second constraint to use x2 instead of x3
def constraint2(m):
    return 3 * m.x2 + m.x4 == 5  # Corrected to use x2

m.c2 = pyo.Constraint(rule=constraint2)

# Define an objective (assuming you want to minimize or maximize x1, x2, x3, or x4)
m.obj = pyo.Objective(expr=m.x2, sense=pyo.minimize)  # Assuming minimization

# Use a solver available in your environment
solver = pyo.SolverFactory('glpk')
solver.solve(m)

# Display solution
m.display()
