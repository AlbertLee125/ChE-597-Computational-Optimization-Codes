import pyomo.environ as pyo

def build_model():
    m = pyo.ConcreteModel()
    
    # Sets of vertices and edges
    V = {0,1,2,3,4,5}
    E = {(0,1), (0,2), (1,2), (1,3), (2,1), (2,4), (3,2), (3,5), (4,3), (4,5)}

    # Cost of each edge
    C = {(0,1):16, (0,2):13, (1,2):10, (1,3):12, (2,1):4, (2,4):14, (3,2):9, (3,5):20, (4,3):7, (4,5):4}

    m.V = pyo.Set(initialize=V)
    m.E = pyo.Set(initialize=E)
    m.c = pyo.Param(m.E, initialize=C)

    m.f = pyo.Var(m.E, domain=pyo.NonNegativeReals)  # Flow on each edge
    m.f_star = pyo.Var(domain=pyo.NonNegativeReals)      # The objective variable f*

    # Objective: Maximize f*
    m.obj = pyo.Objective(expr=m.f_star, sense=pyo.maximize)

    # Flow conservation except source and sink
    def flow_conservation_rule(m, v):
        if v not in {0, 5}:  # Correctly skip the source and sink nodes
            return sum(m.f[e] for e in m.E if e[1] == v) == sum(m.f[e] for e in m.E if e[0] == v)
        else:
            return pyo.Constraint.Skip
    m.flow_conservation = pyo.Constraint(m.V, rule=flow_conservation_rule)


    # Total outflow from source equals f_star
    def source_flow_rule(m):
        source_outflow = sum(m.f[(0, j)] for j in m.V if (0, j) in m.E)
        return source_outflow == m.f_star
    m.source_flow = pyo.Constraint(rule=source_flow_rule)


    def sink_flow_rule(m):
        return sum(m.f[e] for e in m.E if e[1] == 5) - sum(m.f[e] for e in m.E if e[0] == 5) - m.f_star == 0
    m.sink_flow = pyo.Constraint(rule=sink_flow_rule)

    # Capacity constraints
    def capacity_rule(m, i, j):
        e = (i, j)
        return m.f[e] <= m.c[e]
    m.capacity = pyo.Constraint(m.E, rule=capacity_rule)

    return m

solver = pyo.SolverFactory('gurobi')
model = build_model()
solver.solve(model)

print("Optimal flow on each edge:")
for e in model.E:
    print(f"{e}: {model.f[e].value:.4f}")

print(f"Maximum flow (f*): {model.f_star.value:.4f}")
