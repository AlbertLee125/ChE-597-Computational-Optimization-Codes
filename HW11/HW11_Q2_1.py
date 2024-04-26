import pyomo.environ as pyo

def setup_cutting_stock_model():
    # Model and parameters setup
    model = pyo.ConcreteModel()
    widths = [25, 35, 40]
    demands = [7, 5, 3]
    roll_width = 100
    max_rolls = 20  # A reasonable upper bound on number of rolls

    # Variables
    # x[i,j] is the number of i-th width rolls cut from j-th stock roll
    model.x = pyo.Var(range(len(widths)), range(max_rolls), within=pyo.NonNegativeIntegers)
    # y[j] is 1 if j-th stock roll is used
    model.y = pyo.Var(range(max_rolls), within=pyo.Binary)

    # Objective: minimize the number of stock rolls used
    model.obj = pyo.Objective(expr=sum(model.y[j] for j in range(max_rolls)), sense=pyo.minimize)

    # Constraints
    # Demand fulfillment for each width
    model.demand_constraints = pyo.ConstraintList()
    for i in range(len(widths)):
        model.demand_constraints.add(sum(model.x[i, j] for j in range(max_rolls)) >= demands[i])

    # Width limitation for each stock roll
    model.width_constraints = pyo.ConstraintList()
    for j in range(max_rolls):
        model.width_constraints.add(sum(model.x[i, j] * widths[i] for i in range(len(widths))) <= roll_width * model.y[j])

    return model

widths = [25, 35, 40]

# Create and solve the model
model = setup_cutting_stock_model()
solver = pyo.SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# Output results
print("Optimal Solution Found:")
for j in range(20):
    if pyo.value(model.y[j]) > 0.5:  # If roll j is used
        print(f"Stock roll {j+1} is used with cuts:")
        for i in range(len(widths)):
            if pyo.value(model.x[i, j]) > 0:
                print(f" - {int(pyo.value(model.x[i, j]))} rolls of width {widths[i]}")
print(f"Total stock rolls used: {pyo.value(model.obj)}")
