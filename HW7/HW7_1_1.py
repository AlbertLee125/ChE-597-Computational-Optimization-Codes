import pyomo.environ as pyo

m= pyo.ConcreteModel()

# SETS
S = ['A','B','C']   # Substances
I = [1,2,3]   # Processes

# PARAMETERS
m.c_fix = pyo.Param(I, initialize={1:10, 2:15, 3:20})   # Fixed cost of process i [100$/hr]
m.c_var = pyo.Param(I, initialize={1:2.5, 2:4, 3:5.5})   # Variable cost of process i [100$/ton raw]

m.x = pyo.Param(I, initialize={1:0.9, 2:0.82, 3:0.95})   # Conversion of process i [*]

m.p_s = pyo.Param(S, initialize={'A':5, 'B':9.5, 'C':-18}) # Price of selling (if negative) or buying (if positive) substance s [100$/ton]

AUB = 16 # Maximum supply of A [ton/hr]
CUB = 10 # Maximum demand of C [ton/hr], case1: 10, case2: 15

BUB = CUB/0.82 # Upper bound of B [ton/hr]


# VARIABLES
m.y = pyo.Var(I, within=pyo.Binary) # If we install process p

m.A = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,AUB)) # The amount of A we buy [ton/hr]
m.B_buy = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,BUB)) # The amount of B we buy [ton/hr]
m.B_prod = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,BUB)) # The amount of B we produce in process 1 [ton/hr]
m.B2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,BUB)) # The amount of B we put in process 2 [ton/hr]
m.B3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,BUB)) # The amount of B we put in process 3 [ton/hr]
m.C2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,CUB)) # The amount of C we produce in process 2 [ton/hr]
m.C3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,CUB)) # The amount of C we produce in process 3 [ton/hr]
m.C = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,CUB)) # The amount of C we produce in total [ton/hr]

# CONSTRAINTS
@m.Constraint()
def exclusive(m):
    return m.y[2] + m.y[3] == 1

@m.Constraint()
def balance_A(m):
    return m.A * m.x[1] == m.B_prod

@m.Constraint()
def balance_B(m):
    return m.B_buy + m.B_prod == m.B2 + m.B3

@m.Constraint()
def balance_C(m):
    return m.C2 + m.C3 == m.C

@m.Constraint()
def balance_C2(m):
    return m.C2 == m.B2 * m.x[2]

@m.Constraint()
def balance_C3(m):
    return m.C3 == m.B3 * m.x[3]

@m.Constraint()
def A_activation(m):
    return m.A <= AUB * m.y[1]

@m.Constraint()
def B_product_activation(m):
    return m.B_prod <= BUB * m.y[1]

@m.Constraint()
def b_act_2(m):
    return m.B2 <= BUB * m.y[2]

@m.Constraint()
def b_act_3(m):
    return m.B3 <= BUB * m.y[3]

@m.Constraint()
def c_act_2(m):
    return m.C2 <= CUB * m.y[2]

@m.Constraint()
def c_act_3(m):
    return m.C3 <= CUB * m.y[3]

@m.Objective()
def obj(m):
    fixed_costs = sum( m.c_fix[i]*m.y[i] for i in I)
    variable_costs = m.c_var[1]*m.A +  m.c_var[2]*m.B2 + m.c_var[3]*m.B3
    prices = m.p_s['A']*m.A + m.p_s['B']*m.B_buy + m.p_s['C']*m.C 

    return fixed_costs + variable_costs + prices

opt = pyo.SolverFactory('gams', solver='cplex')
results = opt.solve(m, tee=True)

for i in [2,3]:
    if pyo.value(m.y[i]) == 1:
        rta = i

print('Build Process:', rta)
print('Buy', round(pyo.value(m.A),1), 'of A')
print('Buy', round(pyo.value(m.B_buy),1), 'of B')
print('Produce', round(pyo.value(m.B_prod),1), 'of B')
print('Put', round(pyo.value(m.B2),1), 'of B in process 2')
print('Put', round(pyo.value(m.B3),1), 'of B in process 3')
print('Produce', round(pyo.value(m.C),1), 'of C')
print('The profit of the operation is:',round(-100*(pyo.value(m.obj)),1),'$/hr')