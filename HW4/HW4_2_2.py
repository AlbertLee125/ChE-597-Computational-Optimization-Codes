import pyomo.environ as pyo

def build_model():
    m = pyo.ConcreteModel()

    V = {0, 1, 2, 3, 4, 5}  # Set of vertices
    E = {(0, 1), (0, 2), (1, 2), (1, 3), (2, 1), (2, 4), (3, 2), (3, 5), (4, 3), (4, 5)}  # Set of edges
    C = {(0, 1): 16, (0, 2): 13, (1, 2): 10, (1, 3): 12, (2, 1): 4, (2, 4): 14, (3, 2): 9, (3, 5): 20, (4, 3): 7, (4, 5): 4}

    m.V = pyo.Set(initialize=V)
    m.E = pyo.Set(initialize=E)
    m.c = pyo.Param(m.E, initialize=C)

    m.x = pyo.Var(m.V, within=pyo.Reals)  # Node potentials
    m.y = pyo.Var(m.E, within=pyo.Binary)  # Edge in cut or not

    m.obj = pyo.Objective(expr=sum(m.c[e]*m.y[e] for e in m.E), sense=pyo.minimize)

    # Cut constraints
    def cut_rule(m, u, v):
        e = (u, v)
        return m.x[u] - m.x[v] + m.y[e] >= 0
    m.cut = pyo.Constraint(m.E, rule=cut_rule)

    # Source-Sink separation constraint
    m.separation = pyo.Constraint(expr= -m.x[0] + m.x[5] >= 1)

    return m

solver = pyo.SolverFactory('gurobi')
model = build_model()
results = solver.solve(model)

print("Min-Cut Edges:")
for e in model.E:
    if pyo.value(model.y[e]) > 0.5:  # Edge is in the cut
        print(e)

print(f"Minimum cut value: {pyo.value(model.obj)}")

