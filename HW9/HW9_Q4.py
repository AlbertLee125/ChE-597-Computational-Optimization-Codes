import pyomo.environ as pyo

# Create a Pyomo m
m = pyo.ConcreteModel()

# Variable definitions with bounds
m.x1 = pyo.Var(bounds=(1, 2), within=pyo.PositiveReals)
m.x2 = pyo.Var(bounds=(0, 1), within=pyo.NonNegativeReals)

# Objective function
m.objective = pyo.Objective(expr=m.x1 + m.x2, sense=pyo.minimize)

# Constraint
def exp_constraint(m):
    return pyo.exp(m.x2 * pyo.sqrt(m.x1 * m.x2) + pyo.log(m.x1)) <= m.x1**2

m.exp_con = pyo.Constraint(rule=exp_constraint)

# Solving the m using BARON
solver = pyo.SolverFactory('gams', solver = 'baron')

results = solver.solve(m, tee=True)  # tee=True for solver output

# Displaying the results
if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print('Solution is optimal')
    print('x1 =', pyo.value(m.x1))
    print('x2 =', pyo.value(m.x2))
elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print('No feasible solution found')
else:
    print('Solver Status:', results.solver.status)
