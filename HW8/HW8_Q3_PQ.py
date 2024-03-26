from pyomo.environ import *

# Model
model_PQ = ConcreteModel()

# Sets for sources, pools, and products
model_PQ.sources = Set(initialize=['crudeA', 'crudeB', 'crudeC'])
model_PQ.pools = Set(initialize=['pool'])
model_PQ.products = Set(initialize=['finalX', 'finalY'])

# Parameters
model_PQ.supply_cost = Param(model_PQ.sources, initialize={'crudeA': 6, 'crudeB': 16, 'crudeC': 10})
model_PQ.supply_sulfur = Param(model_PQ.sources, initialize={'crudeA': 0.03, 'crudeB': 0.01, 'crudeC': 0.02})
model_PQ.product_price = Param(model_PQ.products, initialize={'finalX': 9, 'finalY': 15})
model_PQ.product_max_sulfur = Param(model_PQ.products, initialize={'finalX': 0.025, 'finalY': 0.015})
model_PQ.product_demand = Param(model_PQ.products, initialize={'finalX': 100, 'finalY': 200})

# Variables
model_PQ.crude_to_pool = Var(model_PQ.sources, model_PQ.pools, domain=NonNegativeReals)
model_PQ.pool_to_product = Var(model_PQ.pools, model_PQ.products, domain=NonNegativeReals)
model_PQ.source_to_product = Var(model_PQ.sources, model_PQ.products, domain=NonNegativeReals)

# Objective: Maximize profit
def profit_rule(model):
    income = sum(model.product_price[p] * (sum(model.pool_to_product[pool, p] for pool in model_PQ.pools) + sum(model.source_to_product[s, p] for s in model_PQ.sources)) for p in model_PQ.products)
    cost = sum(model.supply_cost[s] * (sum(model.crude_to_pool[s, pool] for pool in model_PQ.pools) + sum(model.source_to_product[s, p] for p in model_PQ.products)) for s in model_PQ.sources)
    return income - cost
model_PQ.profit = Objective(rule=profit_rule, sense=maximize)

# Constraints
# Pool Balance Constraints
def pool_balance_rule(model, pool):
    return sum(model.crude_to_pool[s, pool] for s in model_PQ.sources) == sum(model.pool_to_product[pool, p] for p in model_PQ.products)
model_PQ.pool_balance = Constraint(model_PQ.pools, rule=pool_balance_rule)

# Product Quality Constraints (for both pool-to-product and direct source-to-product flows)
def product_quality_rule(model, product):
    blended_sulfur = sum(model.supply_sulfur[s] * (model.pool_to_product['pool', product] + model.source_to_product[s, product]) for s in model_PQ.sources)
    total_volume = sum(model.pool_to_product['pool', product] + model.source_to_product[s, product] for s in model_PQ.sources)
    return blended_sulfur <= model.product_max_sulfur[product] * total_volume
model_PQ.product_quality = Constraint(model_PQ.products, rule=product_quality_rule)

# Product Demand Constraints (considering both pool-to-product and direct source-to-product flows)
def product_demand_rule(model, product):
    total_production = sum(model.pool_to_product[pool, product] for pool in model_PQ.pools) + sum(model.source_to_product[s, product] for s in model_PQ.sources)
    return total_production <= model.product_demand[product]
model_PQ.product_demand = Constraint(model_PQ.products, rule=product_demand_rule)

# Solve the model
solver = SolverFactory('ipopt')  # Using IPOPT solver, suitable for NLP problems
result = solver.solve(model_PQ, tee=True)

# Display results
model_PQ.pprint()
