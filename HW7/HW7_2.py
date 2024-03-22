import pyomo.environ as pyo

# Define the data and parameters
demand = [10, 40, 20, 5, 5, 15]
fixed_cost = 50
production_cost = [1, 3, 3, 1, 1, 1]
holding_cost = 2
production_limit = 25
T = len(demand)  # Number of periods

# Initialize the model
m = pyo.ConcreteModel()

# Define the decision variables
m.production = pyo.Var(range(T), within= pyo.NonNegativeReals)
m.inventory =  pyo.Var(range(T), within= pyo.NonNegativeReals)
m.activate_production =  pyo.Var(range(T), within= pyo.Binary)

# Define the objective function
def objective_rule(m):
    return sum(production_cost[t] * m.production[t] + holding_cost * m.inventory[t] + 
               fixed_cost * m.activate_production[t] for t in range(T))

m.objective =  pyo.Objective(rule=objective_rule, sense= pyo.minimize)

# Define the constraints

# Demand satisfaction constraint
def demand_satisfaction_rule(m, t):
    if t == 0:
        return m.production[t] - demand[t] == m.inventory[t]
    return m.inventory[t-1] + m.production[t] - demand[t] == m.inventory[t]

m.demand_satisfaction =  pyo.Constraint(range(T), rule=demand_satisfaction_rule)

# Production limit constraint
def production_limit_rule(m, t):
    return m.production[t] <= production_limit * m.activate_production[t]

m.production_limit =  pyo.Constraint(range(T), rule=production_limit_rule)

# Inventory non-negativity is ensured by variable definition

# Solve the model
solver =  pyo.SolverFactory('gurobi')
results = solver.solve(m, tee=True)

# Extract the solution
production_plan = [m.production[t].value for t in range(T)]
inventory_levels = [m.inventory[t].value for t in range(T)]
activate_production_decisions = [m.activate_production[t].value for t in range(T)]

# Print the solution
print('Production plan:', production_plan)
print('Inventory levels:', inventory_levels)
print('Activate production decisions:', activate_production_decisions)
print('Objective function value:', m.objective())