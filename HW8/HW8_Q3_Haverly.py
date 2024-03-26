from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Sets
model.s = Set(initialize=['crudeA', 'crudeB', 'crudeC'], doc='Supplies (crudes)')
model.f = Set(initialize=['finalX', 'finalY'], doc='Final products')
model.i = Set(initialize=['Pool', 'CrudeC'], doc='Intermediate sources for final products')
model.poolin = Set(initialize=['crudeA', 'crudeB'], doc='Crudes going into pool tank')

# Parameters
data_S = {
    'crudeA': {'price': 6, 'sulfur': 3},
    'crudeB': {'price': 16, 'sulfur': 1},
    'crudeC': {'price': 10, 'sulfur': 2}
}

data_F = {
    'finalX': {'price': 9, 'sulfur': 2.5, 'demand': 100},
    'finalY': {'price': 15, 'sulfur': 1.5, 'demand': 200}
}

model.sulfur_content = Param(model.s, initialize={s: data_S[s]['sulfur'] for s in model.s}, doc='Supply quality in percent')
model.req_sulfur = Param(model.f, initialize={f: data_F[f]['sulfur'] for f in model.f}, doc='Required max sulfur content (percentage)')
model.demand = Param(model.f, initialize={f: data_F[f]['demand'] for f in model.f}, doc='Final product demand')

# Variables
model.crude = Var(model.s, domain=NonNegativeReals, doc='Amount of crudes being used')
model.stream = Var(model.i, model.f, domain=NonNegativeReals, doc='Streams')
model.q = Var(domain=NonNegativeReals, doc='Pool quality')
model.profit = Var(doc='Total profit')
model.cost = Var(doc='Total costs')
model.income = Var(doc='Total income')
model.final = Var(model.f, domain=NonNegativeReals, bounds=(0, None), doc='Amount of final products sold')

# Objective
def profit_rule(model):
    return model.income - model.cost
model.profit = Objective(rule=profit_rule, sense=maximize, doc='Profit equation')

# Constraints
def cost_rule(model):
    return model.cost == sum(data_S[s]['price'] * model.crude[s] for s in model.s)
model.cost_constraint = Constraint(rule=cost_rule, doc='Cost equation')

def income_rule(model):
    return model.income == sum(data_F[f]['price'] * model.final[f] for f in model.f)
model.income_constraint = Constraint(rule=income_rule, doc='Income equation')

def blend_rule(model, f):
    return model.final[f] == sum(model.stream[i, f] for i in model.i)
model.blend_constraint = Constraint(model.f, rule=blend_rule, doc='Blending of final products')

def pool_bal_rule(model):
    return sum(model.crude[s] for s in model.poolin) == sum(model.stream['Pool', f] for f in model.f)
model.pool_bal_constraint = Constraint(rule=pool_bal_rule, doc='Pool tank balance')

def crudeC_bal_rule(model):
    return model.crude['crudeC'] == sum(model.stream['CrudeC', f] for f in model.f)
model.crudeC_bal_constraint = Constraint(rule=crudeC_bal_rule, doc='Balance for crudeC')

def pool_qual_bal_rule(model):
    return model.q * sum(model.stream['Pool', f] for f in model.f) == sum(model.sulfur_content[s] * model.crude[s] for s in model.poolin)
model.pool_qual_bal_constraint = Constraint(rule=pool_qual_bal_rule, doc='Pool quality balance')

def blend_qual_bal_rule(model, f):
    return model.q * model.stream['Pool', f] + model.sulfur_content['crudeC'] * model.stream['CrudeC', f] <= model.req_sulfur[f] * sum(model.stream[i, f] for i in model.i)
model.blend_qual_bal_constraint = Constraint(model.f, rule=blend_qual_bal_rule, doc='Quality balance for blending')

# Solver
solver = SolverFactory('ipopt')
model.q.set_value(1)  # Set initial value for q
result = solver.solve(model, tee=True)

# Display results
model.display()