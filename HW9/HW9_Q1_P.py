from pyomo.environ import *

# P-Formulation Model with McCormick Envelopes
model_P = ConcreteModel()

# Sets for sources, pools, and products
model_P.sources = Set(initialize=['crudeA', 'crudeB', 'crudeC'])
model_P.pools = Set(initialize=['pool'])
model_P.products = Set(initialize=['finalX', 'finalY'])

# Parameters
model_P.supply_cost = Param(model_P.sources, initialize={'crudeA': 6, 'crudeB': 16, 'crudeC': 10})
model_P.supply_sulfur = Param(model_P.sources, initialize={'crudeA': 0.03, 'crudeB': 0.01, 'crudeC': 0.02})
model_P.product_price = Param(model_P.products, initialize={'finalX': 9, 'finalY': 15})
model_P.product_max_sulfur = Param(model_P.products, initialize={'finalX': 0.025, 'finalY': 0.015})
model_P.product_demand = Param(model_P.products, initialize={'finalX': 100, 'finalY': 200})

# Variables
model_P.crude_to_pool = Var(model_P.sources, model_P.pools, domain=NonNegativeReals)
model_P.pool_to_product = Var(model_P.pools, model_P.products, domain=NonNegativeReals)
model_P.pool_sulfur = Var(model_P.pools, domain=NonNegativeReals)

# New Variables for Bilinear Terms
model_P.crude_pool_sulfur = Var(model_P.sources, model_P.pools, domain=NonNegativeReals)
model_P.product_sulfur_content = Var(model_P.pools, model_P.products, domain=NonNegativeReals)

# Objective: Maximize profit
def profit_rule(model):
    income = sum(model.product_price[p] * sum(model.pool_to_product[pool, p] for pool in model_P.pools) for p in model_P.products)
    cost = sum(model.supply_cost[s] * sum(model.crude_to_pool[s, pool] for pool in model_P.pools) for s in model_P.sources)
    return income - cost
model_P.profit = Objective(rule=profit_rule, sense=maximize)

# Constraints
# Pool Balance
def pool_balance_rule(model, pool):
    return sum(model.crude_to_pool[s, pool] for s in model_P.sources) == sum(model.pool_to_product[pool, p] for p in model_P.products)
model_P.pool_balance = Constraint(model_P.pools, rule=pool_balance_rule)

# Modified Pool Sulfur Content Constraint with McCormick
def pool_sulfur_rule(model, pool):
    return sum(model.crude_pool_sulfur[s, pool] for s in model_P.sources) == model.pool_sulfur[pool] * sum(model.pool_to_product[pool, p] for p in model_P.products)
model_P.pool_sulfur_constraint = Constraint(model_P.pools, rule=pool_sulfur_rule)

# McCormick Envelope Constraints for crude_pool_sulfur
# These constraints should be formulated based on the bounds of the variables involved
# [Add McCormick Envelope constraints here]

# Modified Product Sulfur Constraint with McCormick
def product_sulfur_rule(model, product):
    return sum(model.product_sulfur_content[pool, product] for pool in model_P.pools) <= model.product_max_sulfur[product] * sum(model.pool_to_product[pool, product] for pool in model_P.pools)
model_P.product_sulfur_constraint = Constraint(model_P.products, rule=product_sulfur_rule)

# McCormick Envelope Constraints for product_sulfur_content
# These constraints should be formulated based on the bounds of the variables involved
# [Add McCormick Envelope constraints here]

# Product Demand Constraint
def product_demand_rule(model, product):
    return sum(model.pool_to_product[pool, product] for pool in model_P.pools) <= model.product_demand[product]
model_P.product_demand_constraint = Constraint(model_P.products, rule=product_demand_rule)

# Solve the model
solver = SolverFactory('ipopt')
solver.solve(model_P, tee=True)

model_P.pprint()
print('Profit: ', model_P.profit())
print('Crude to Pool:')
for s in model_P.sources:
    for pool in model_P.pools:
        print(s, pool, model_P.crude_to_pool[s, pool]())
print('Pool to Product:')
for pool in model_P.pools:
    for p in model_P.products:
        print(pool, p, model_P.pool_to_product[pool, p]())
print('Pool Sulfur:')
for pool in model_P.pools:
    print(pool, model_P.pool_sulfur[pool]())
