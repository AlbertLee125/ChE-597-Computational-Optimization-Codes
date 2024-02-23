import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.x1 = pyo.Var(domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(domain=pyo.NonNegativeReals)

def objective_rule(m):
    return 3/(m.x1 + m.x2) + m.x2 + pyo.exp(m.x1) + (m.x1 - m.x2)**2

m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

def inequality_constraint_rule(m, i):
    if i == 1:
        return m.x1**2 + m.x2**2 <= 2
    else:
        return m.x1 - m.x2 <= 1
    
m.inequality_constraint = pyo.Constraint([1, 2], rule=inequality_constraint_rule)

solver = pyo.SolverFactory('ipopt')
solver.solve(m)

print(f"Optimal solution: x1 = {m.x1.value:.4f}, x2 = {m.x2.value:.4f}")
print(f"Optimal objective: {m.objective():.4f}")
# Optimal solution: x1 = 0.6179, x2 = 0.8317
# Optimal objective: 4.8020