import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.x1 = pyo.Var(bounds=[0, 3], domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(bounds=[0, 5], domain=pyo.NonNegativeReals)

m.obj = pyo.Objective(expr=m.x1 - 2*m.x2, sense=pyo.minimize)

m.con1 = pyo.Constraint(expr= -1 * (m.x1)**2 + (m.x2)**2 + 4* (m.x2) * (m.x1) <= 7)

# Solve
solver = pyo.SolverFactory('ipopt')
solver.solve(m)

# Results rounded 4 digits
print('Results:')
print('x1:', round(pyo.value(m.x1), 4))
print('x2:', round(pyo.value(m.x2), 4))
print('Objective:', round(pyo.value(m.obj), 4))