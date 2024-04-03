import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.x1 = pyo.Var(bounds=(0, 3), domain=pyo.NonNegativeReals)
m.x2 = pyo.Var(bounds=(0, 5), domain=pyo.NonNegativeReals)

m.obj = pyo.Objective(expr=m.x1 - 2*m.x2, sense=pyo.minimize)

m.constraint = pyo.Constraint(expr=2*m.x1**2 + 4*m.x1*m.x2 + 2*m.x2**2 - 9*m.x1 - 5*m.x2 <= 7)

solver = pyo.SolverFactory('gurobi')  
results = solver.solve(m, tee=True)

# Display the results
print("Status:", results.solver.status)
print("Termination Condition:", results.solver.termination_condition)

# Display the value of the variables at the optimum
if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("Optimal Solution:")
    print("Objective value =", round(pyo.value(m.obj),4))
    print("x1 =", round(pyo.value(m.x1),4))
    print("x2 =", round(pyo.value(m.x2),4))
else:
    print("No optimal solution found.")
