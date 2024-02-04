import pyomo.environ as pyo

m = pyo.ConcreteModel()

# Sets
K = [1, 2, 3, 4]
I = ['H1', 'H2']
J = ['C1', 'C2']

# Parameters
QH = {(i,k):0 for i in I for k in K}

QH['H1',2] = 60
QH['H1',3] = 160
QH['H1',4] = 60

QH['H2',3] = 320
QH['H2',4] = 120

QC = {(j,k):0 for j in J for k in K}

QC['C1',1] = 30
QC['C1',2] = 90
QC['C1',3] = 240

QC['C2',3] = 117
QC['C2',4] = 78

# Variables
m.R = pyo.Var(I,K, within=pyo.NonNegativeReals)
m.RM = pyo.Var(K, within=pyo.NonNegativeReals)
m.QM = pyo.Var(K, within=pyo.NonNegativeReals)
m.QN = pyo.Var(K, within=pyo.NonNegativeReals)
m.Q = pyo.Var(I,J,K, within=pyo.NonNegativeReals)
m.QIN = pyo.Var(I,K, within=pyo.NonNegativeReals)
m.QMJ = pyo.Var(J,K, within=pyo.NonNegativeReals)

# Constraints
for i in I:
    m.R[i,4].fix(0)

m.RM[4].fix(0)

# No H1 C1 match, Temperature Feasibility
for k in K:
    if k != 1:
        m.QM[k].fix(0)
    if k != 4:
        m.QN[k].fix(0)
    m.Q['H1','C1',k].fix(0)

@m.Constraint(I,K)
def heatik(m,i,k):
    if k == 1:
        return m.R[i,k] + sum(m.Q[i,j,k] for j in J) + m.QIN[i,k] == QH[i,k]
    else:
        return m.R[i,k] - m.R[i,k-1] + sum(m.Q[i,j,k] for j in J) + m.QIN[i,k] == QH[i,k]

@m.Constraint(K)
def heatmk(m,k):
    if k == 1:
        return m.RM[k] + sum(m.QMJ[j,k] for j in J) - m.QM[k] == 0
    else:
         return m.RM[k] - m.RM[k-1] + sum(m.QMJ[j,k] for j in J) - m.QM[k] == 0


@m.Constraint(J,K)
def heatjk(m,j,k):
    return sum(m.Q[i,j,k] for i in I) + m.QMJ[j,k] == QC[j,k]

@m.Constraint(K)
def heatnk(m,k):
    return sum(m.QIN[i,k] for i in I) - m.QN[k] == 0

# Objective
@m.Objective(sense = pyo.minimize)
def obj(m):
    return sum(m.QM[k] + m.QN[k] for k in K) 

opt = pyo.SolverFactory('gams', solver='cplex')
results = opt.solve(m, tee=False,
                    add_options=[
                        'option reslim = 200;'
                        'option optcr = 0.0;']) # Set solver options

# Print results
print('Utility Consumption [kW]:',m.obj())
print('Cooling Utility Required [kW]:', sum(m.QIN[i,k]() for i in I for k in K))
print('Heating Utility Required [kW]:', sum(m.QMJ[j,k]() for j in J for k in K))

# After solving the model
for v in m.component_objects(pyo.Var, active=True):
    print("Variable",v)
    var_object = getattr(m, str(v))
    for index in var_object:
        print(" ",index, var_object[index].value)
