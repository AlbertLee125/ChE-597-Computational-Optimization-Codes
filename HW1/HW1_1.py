import pyomo.environ as pyo

def model():
    m = pyo.ConcreteModel()
    m.i = pyo.Set(initialize=[1, 2, 3])  # Designer Number
    m.j = pyo.Set(initialize=[1, 2, 3, 4])  # Project Number
    H = {1: 70, 2: 50, 3: 85, 4: 35}  # Hours required for each project
    m.x = pyo.Var(m.i, m.j, domain=pyo.NonNegativeReals)

    # Each Designer has a maximum of 80 hours
    def constraint_rule1(m, i):
        return sum(m.x[i, j] for j in m.j) <= 80
    m.cons1 = pyo.Constraint(m.i, rule=constraint_rule1)

    # Each Project has its own minimum hours required
    def constraint_rule2(m, j):
        return sum(m.x[i, j] for i in m.i) >= H[j]
    m.cons2 = pyo.Constraint(m.j, rule=constraint_rule2)

    # Objective function
    def obj(m):
        return sum(m.x[i, j] * capabilities[i, j] for i in m.i for j in m.j)
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)

    return m

# Define the capabilities matrix as a dictionary
capabilities = {
    (1, 1): 90, (1, 2): 80, (1, 3): 10, (1, 4): 50,
    (2, 1): 60, (2, 2): 70, (2, 3): 50, (2, 4): 65,
    (3, 1): 70, (3, 2): 40, (3, 3): 80, (3, 4): 85
}

if __name__ == '__main__':
    m = model()
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(m)
    # Instead of m.display(), print the results
    for i in m.i:
        for j in m.j:
            print('Designer', i, 'Project', j, 'Hours:', pyo.value(m.x[i, j]))

    print('Maximum Scoring:', pyo.value(m.obj))
