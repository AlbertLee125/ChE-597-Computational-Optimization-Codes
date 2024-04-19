# Bikini manufacturing problem

# Global Minimum is a company specializing in the manufacture of bikini swimming suits, facing the challenge of highly seasonal demand patterns.
# The expected demands for the upcoming year, distributed across four quarters, vary significantly, necessitating careful planning to meet these demands efficiently. 
# The objective is to devise an inventory management strategy that enables Global Minimum to fulfill its projected quarterly demand for bikini swimming suits while minimizing inventory holding costs. 

# Reference: Rardin, Ronald L. Optimization in operations research. Vol. 166. Upper Saddle River, NJ: Prentice Hall, 1998.
# Source: OIOR Exercise 4.20


"""
Style Guide:

- Use short and descriptive names for variables, parameters, and constraints.
- Use comments to describe the purpose of the model, sets, parameters, variables, and constraints.
- Always use the doc attribute to provide a short description of the model, sets, parameters, variables, and constraints.
- Always indicate Sets with Capitalized names and variables with lowercase names.
- Always use `import pyomo.environ as pyo` to import Pyomo.
- Use different names for each set, constraint, variable, and parameter.
- Always make the parameters Mutable, i.e. set `mutable=True`.
- Always name the model as `model`.
"""

import pyomo.environ as pyo

# Create the Pyomo model
model = pyo.ConcreteModel()

# Set of quarters
model.T = pyo.Set(initialize=[1, 2, 3, 4], doc='Quarters')

# Parameters
model.demand = pyo.Param(model.T, initialize={1: 2800, 2: 500, 3: 100, 4: 1200}, mutable=True, doc='Demand for each quarter') # Adjusted the demand for Last quarter 4 to 1200
model.capacity = pyo.Param(model.T, initialize={1: 1200, 2: 1200, 3: 800, 4: 1200}, mutable=True, doc='Production capacity for each quarter') # Adjust the third quarter into 800
model.holding_cost = pyo.Param(model.T, initialize={1: 15, 2: 15, 3: 15, 4: 15}, mutable=True, doc = 'Holding cost for each quarter')
model.holding_cost_initial = pyo.Param(initialize=15, doc='Holding cost for initial inventory', mutable=True)

# Variables
model.production = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Production quantities')
model.inventory = pyo.Var(model.T, domain=pyo.NonNegativeReals, doc='Inventory levels at the end for each quarter')
model.initial_inventory = pyo.Var(domain=pyo.NonNegativeReals, doc='Initial Inventory')

# Objective: Minimize total holding costs
model.cost = pyo.Objective(expr=model.holding_cost_initial * model.initial_inventory + 
                           sum(model.holding_cost[t] * model.inventory[t] for t in model.T), sense=pyo.minimize)

# Constraints
def production_constraint(model, t):
    return model.production[t] <= model.capacity[t]
model.prod_con = pyo.Constraint(model.T, rule=production_constraint, doc='Production capacity constraint')

def inventory_balance(model, t):
    if t == 1:
        return model.initial_inventory + model.production[t] - model.demand[t] == model.inventory[t]
    else:
        return model.inventory[t-1] + model.production[t] - model.demand[t] == model.inventory[t]
model.inv_bal = pyo.Constraint(model.T, rule=inventory_balance, doc='Inventory balance constraint')

def end_cycle_inventory(model):
    return model.inventory[model.T.last()] >= model.initial_inventory
model.end_cycle_con = pyo.Constraint(rule=end_cycle_inventory, doc='End cycle inventory constraint')

# Deactivate the original inventory balance constraint
model.inv_bal.deactivate()

# Add slack variables for each quarter to handle possible demand overflows
model.slack = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0, 300), initialize=0, doc="Slack variables to manage excess demand")

# Define a new inventory balance constraint that includes slack variables
def inventory_balance_with_slack(model, t):
    if t == 1:
        return model.initial_inventory + model.production[t] + model.slack[t] - model.demand[t] == model.inventory[t]
    else:
        return model.inventory[t-1] + model.production[t] + model.slack[t] - model.demand[t] == model.inventory[t]
