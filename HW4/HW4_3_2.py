import pyomo.environ as pyo

# Assuming the same A matrix from the police problem
A = [[4, -10, -10],  # Loss when terrorists attack site 1
     [-8, 5, -8],  # Loss when terrorists attack site 2
     [-12, -12, 9]]  # Loss when terrorists attack site 3

model = pyo.ConcreteModel()

# Defining the variables
model.u = pyo.Var(range(3), within=pyo.NonNegativeReals)  # Probabilities for attacking each site by terrorists
model.w = pyo.Var()  # Maximum loss to minimize

# Objective: Minimize w
model.objective = pyo.Objective(expr=model.w, sense=pyo.minimize)

# Constraints
def loss_constraints(model, j):
    # u^T A - w <= 0, rewritten as w - u^T A >= 0 for each police strategy
    return model.w - sum(A[i][j] * model.u[i] for i in range(3)) >= 0

model.loss_constraints = pyo.Constraint(range(3), rule=loss_constraints)

def probability_constraint(model):
    # The sum of probabilities should be 1
    return sum(model.u[i] for i in range(3)) == 1

model.probability_constraint = pyo.Constraint(rule=probability_constraint)

# Solve the model
solver = pyo.SolverFactory('gurobi')  
solver.solve(model)

# Print the results
print("Optimal probabilities for attacking each site:")
for i in range(3):
    print(f"Site {i+1}: {model.u[i].value:.4f}")

print(f"Maximum expected loss (w): {model.w.value:.4f}")
