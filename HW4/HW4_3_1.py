import pyomo.environ as pyo

# Given A matrix (utility matrix), assuming a 3x3 matrix for three sites
A = [[4, -10, -10],  # Utility when police patrol site 1
     [-8, 5, -8],  # Utility when police patrol site 2
     [-12, -12, 9]]  # Utility when police patrol site 3

model = pyo.ConcreteModel()

# Defining the variables
model.x = pyo.Var(range(3), within=pyo.NonNegativeReals)  # Probabilities for patrolling each site
model.z = pyo.Var()  # Minimum utility to maximize

# Objective: Maximize z
model.objective = pyo.Objective(expr=model.z, sense=pyo.maximize)

# Constraints
def utility_constraints(model, i):
    # Ax - z <= 0, rewritten as z - Ax <= 0 for each site
    return model.z - sum(A[i][j] * model.x[j] for j in range(3)) <= 0

model.utility_constraints = pyo.Constraint(range(3), rule=utility_constraints)

def probability_constraint(model):
    # The sum of probabilities should be 1
    return sum(model.x[j] for j in range(3)) == 1

model.probability_constraint = pyo.Constraint(rule=probability_constraint)

# Solve the model
solver = pyo.SolverFactory('gurobi')  
solver.solve(model)

# Print the results
print("Optimal probabilities for patrolling each site:")
for j in range(3):
    print(f"Site {j+1}: {model.x[j].value:.4f}")

print(f"Minimum expected utility (z): {model.z.value:.4f}")
