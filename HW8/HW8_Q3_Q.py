from pyomo.environ import *

# Model
model_Q = ConcreteModel()

# Sets for sources and products
model_Q.sources = Set(initialize=['crudeA', 'crudeB', 'crudeC'])
model_Q.products = Set(initialize=['finalX', 'finalY'])

# Parameters
model_Q.supply_cost = Param(model_Q.sources, initialize={'crudeA': 6, 'crudeB': 16, 'crudeC': 10})
model_Q.supply_sulfur = Param(model_Q.sources, initialize={'crudeA': 0.03, 'crudeB': 0.01, 'crudeC': 0.02})
model_Q.product_price = Param(model_Q.products, initialize={'finalX': 9, 'finalY': 15})
model_Q.product_max_sulfur = Param(model_Q.products, initialize={'finalX': 0.025, 'finalY': 0.015})
model_Q.product_demand = Param(model_Q.products, initialize={'finalX': 100, 'finalY': 200})

# Variables
model_Q.source_to_product = Var(model_Q.sources, model_Q.products, domain=NonNegativeReals)

# Objective: Maximize profit
def profit_rule(model):
    income = sum(model.product_price[p] * sum(model.source_to_product[s, p] for s in model_Q.sources) for p in model_Q.products)
    cost = sum(model.supply_cost[s] * sum(model.source_to_product[s, p] for p in model_Q.products) for s in model_Q.sources)
    return income - cost
model_Q.profit = Objective(rule=profit_rule, sense=maximize)

# Constraints
# Product Quality Constraints
def product_quality_rule(model, product):
    total_sulfur = sum(model.source_to_product[s, product] * model.supply_sulfur[s] for s in model_Q.sources)
    total_volume = sum(model.source_to_product[s, product] for s in model_Q.sources)
    return total_sulfur <= model.product_max_sulfur[product] * total_volume
model_Q.product_quality = Constraint(model_Q.products, rule=product_quality_rule)

# Product Demand Constraints
def product_demand_rule(model, product):
    return sum(model.source_to_product[s, product] for s in model_Q.sources) <= model.product_demand[product]
model_Q.product_demand = Constraint(model_Q.products, rule=product_demand_rule)

# Solve the model
solver = SolverFactory('ipopt')  # IPOPT is recommended for NLP problems, ensure it is installed and available
result = solver.solve(model_Q, tee=True)

# Display results
model_Q.pprint()

# Display results
print('Profit: ', model_Q.profit())
print('Source to Product:')
for s in model_Q.sources:
    for p in model_Q.products:
        print(s, p, model_Q.source_to_product[s, p]())

