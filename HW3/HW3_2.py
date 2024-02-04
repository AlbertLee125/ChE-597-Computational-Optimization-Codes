import pyomo.environ as pyo

# Define the model
m = pyo.ConcreteModel()

# Define the variables within the non-negative real numbers
m.x1 = pyo.Var(within=pyo.NonNegativeReals)
m.x2 = pyo.Var(within=pyo.NonNegativeReals)

# Define the constraints
m.constraint1 = pyo.Constraint(expr=m.x1 - m.x2 >= -2)
m.constraint2 = pyo.Constraint(expr=m.x1 + m.x2 >= 1)

# Constraints 3 and 4 (x1 >= 0 and x2 >= 0) are already defined by the non-negativity of the variables.
# Define the objective function
m.objective = pyo.Objective(expr=m.x1 + 2*m.x2, sense=pyo.maximize)

# Define the solver
solver = pyo.SolverFactory('glpk')  

# Solve the model
solution = solver.solve(m, tee=True)

# Display results
m.x1.display()
m.x2.display()
