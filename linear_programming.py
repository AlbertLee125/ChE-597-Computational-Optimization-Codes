import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.a = pyo.Var(within=pyo.Reals, initialize=0.0)
m.b = pyo.Var(within=pyo.Reals, initialize=0.0)
m.alpha = pyo.Var(within=pyo.Reals, initialize=0.0)

m.obj = pyo.Objective(expr=+ 4 * m.a + 1 * m.b + m.alpha, sense=pyo.maximize)

m.constraint1 = pyo.Constraint(expr= + 5 * m.a + 1 * m.b + m.alpha <= -0.542)
m.constraint2 = pyo.Constraint(expr= + 5 * m.a + 2 * m.b + m.alpha <= -0.818)
m.constraint3 = pyo.Constraint(expr= + 4 * m.a + 1 * m.b + m.alpha <= -0.639)
m.constraint4 = pyo.Constraint(expr= + 4 * m.a + 2 * m.b + m.alpha <= -0.942)

solver = pyo.SolverFactory('gurobi')
solver.solve(m)

print('alpha:', pyo.value(m.alpha))
print('p1:', pyo.value(m.a))
print('p2:', pyo.value(m.b))
print('obj:', pyo.value(m.obj))