import pyomo.environ as pyo

def build_stochastic_farmers_lagrangian():
    m = pyo.ConcreteModel()
    I = [1, 2, 3]  # Scenarios

    # Decision variables initialization with bounds
    m.x1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 500))
    m.x2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 500))
    m.x3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 500))
    m.y1 = pyo.Var(I, within=pyo.NonNegativeReals)
    m.y2 = pyo.Var(I, within=pyo.NonNegativeReals)
    m.w1 = pyo.Var(I, within=pyo.NonNegativeReals)
    m.w2 = pyo.Var(I, within=pyo.NonNegativeReals)
    m.w3 = pyo.Var(I, within=pyo.NonNegativeReals)
    m.w4 = pyo.Var(I, within=pyo.NonNegativeReals)

    # Initialize Lagrange Multipliers
    m.lambda_wheat = pyo.Var(I, within=pyo.Reals, initialize=0)
    m.lambda_corn = pyo.Var(I, within=pyo.Reals, initialize=0)

    # Define the objective function with safe multiplier updates
    def obj_rule(m):
        return sum([150*m.x1, 230*m.x2, 260*m.x3]) + \
               sum([1/3 * (238*m.y1[i] - 170*m.w1[i] + 210*m.y2[i] - 150*m.w2[i] - 36*m.w3[i] - 10*m.w4[i] + \
                          m.lambda_wheat[i] * (3 * m.x1 + m.y1[i] - m.w1[i] - 200) + \
                          m.lambda_corn[i] * (3.6 * m.x2 + m.y2[i] - m.w2[i] - 240))
                    for i in I])
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Define constraints
    m.cons1 = pyo.Constraint(expr=m.x1 + m.x2 + m.x3 <= 500)  # Total land constraint
    return m

def update_multipliers(model, step_size=0.01):
    for i in model.y1:
        if model.x1.value is not None and model.y1[i].value is not None and model.w1[i].value is not None:
            model.lambda_wheat[i].value += step_size * (3 * model.x1.value + model.y1[i].value - model.w1[i].value - 200)
        if model.x2.value is not None and model.y2[i].value is not None and model.w2[i].value is not None:
            model.lambda_corn[i].value += step_size * (3.6 * model.x2.value + model.y2[i].value - model.w2[i].value - 240)

def solve_farmers_problem():
    model = build_stochastic_farmers_lagrangian()
    solver = pyo.SolverFactory('gurobi')
    for iteration in range(10):
        result = solver.solve(model, tee=True)
        if result.solver.status == pyo.SolverStatus.ok:
            update_multipliers(model)
            print("Iteration:", iteration, "Objective:", pyo.value(model.obj))
        else:
            print("Solver reported a problem: ", result.solver.termination_condition)
    return model

# Run the model
m = solve_farmers_problem()
