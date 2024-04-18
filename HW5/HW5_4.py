import pyomo.environ as pyo
import numpy as np

m = pyo.ConcreteModel()

m.r = pyo.Var(domain=pyo.NonNegativeReals, initialize=1.0)
m.h = pyo.Var(domain=pyo.NonNegativeReals, initialize=1.0)

def Volume(m):
    return np.pi * m.r**2 * m.h == 25

m.Volume = pyo.Constraint(rule=Volume)

def Cost(m):
    return 190 * np.pi * m.r**2 + 260 * np.pi * m.r**2 + 150 * 2 * np.pi * m.r * m.h

m.Cost = pyo.Objective(rule=Cost, sense=pyo.minimize)

solver = pyo.SolverFactory('gams,')
solver.solve(m)

print(f"Optimal radius: {m.r.value:.4f} m")
print(f"Optimal height: {m.h.value:.4f} m")
print(f"Minimum cost: $ {m.Cost():.2f}")
