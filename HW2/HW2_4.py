import pyomo.environ as pyo

def build_model():
    m = pyo.ConcreteModel()

    # Initialize Sets directly in the model
    m.nodes = pyo.Set(initialize={'s1', 's2', 's3', 't1', 't2', 't3', 'x', 'y'})
    m.edges = pyo.Set(dimen=2, initialize={('s1', 't1'), ('s1','x'), ('s2', 'x'), ('s3','x'), ('s3', 't3'), ('x', 'y'), ('y', 't1'), ('y', 't2'), ('y', 't3')})
    m.commodities = pyo.Set(initialize={'d1', 'd2', 'd3'})

    # Initialize Parameters directly in the model
    m.demand = pyo.Param(m.commodities, initialize={'d1':9, 'd2':2, 'd3':3})
    m.capacity = pyo.Param(m.edges, initialize={('s1', 't1'):7, ('s1','x'):5, ('s2', 'x'):3, ('s3','x'):5, ('s3', 't3'):5, ('x', 'y'):14, ('y', 't1'):3, ('y', 't2'):3, ('y', 't3'):4})
    m.cost = pyo.Param(m.edges, m.commodities, initialize={('s1', 't1'):3, ('s1','x'):1, ('s2', 'x'):2, ('s3','x'):2, ('s3', 't3'):5, ('x', 'y'):2, ('y', 't1'):1, ('y', 't2'):2, ('y', 't3'):2})

    # Variables: the flow of commodity k through edge e
    m.flow = pyo.Var(m.edges, m.commodities, domain=pyo.NonNegativeReals)

    # Objective: minimize the cost
    def objective_rule(m):
        return sum(m.flow[e,c]* m.cost[e,c] for e in m.edges for c in m.commodities)
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Capacity Constraints: should only be defined for each edge
    def capacity_rule(m, e):
        return sum(m.flow[e, c] for c in m.commodities) <= m.capacity[e]
    m.capacity_constraints = pyo.Constraint(m.edges, rule=capacity_rule)

    # Flow Conservation Constraints: a single rule for all nodes
    def flow_rule(m, n, c):
        inflow = sum(m.flow[(i,n), c] for i in m.nodes if (i,n) in m.edges)
        outflow = sum(m.flow[(n,j), c] for j in m.nodes if (n,j) in m.edges)
        if n[0] == 's':  # source nodes
            return (outflow - inflow) == m.demand[c]
        elif n[0] == 't':  # sink nodes
            return (inflow - outflow) == m.demand[c]
        else:  # intermediate nodes
            return inflow == outflow
    m.flow_conservation = pyo.Constraint(m.nodes, m.commodities, rule=flow_rule)

    return m

if __name__ == "__main__":
    model = build_model()
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)
    model.display()