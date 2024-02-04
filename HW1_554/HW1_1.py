import pyomo.environ as pyo

# Create the data reconcilation model
m = pyo.ConcreteModel()

# Define the variables for the flow rates, with an index from 1 to 28
m.F = pyo.Var(range(1, 29), within=pyo.NonNegativeReals)

# Measured flow rates from the problem statement
measured_flows = {
    1: 0.90, 2: 1.0, 3: 112.82, 4: 109.95, 5: 53.27, 
    6: 113.27, 7: 2.32, 8: 165.0, 9: 0.86, 10: 52.41,
    11: 15.0, 12: 67.30, 13: 111.27, 14: 91.86, 15: 61.0,
    16: 23.64, 17: 33.0, 18: 16.23, 19: 8.0, 20: 10.50,
    21: 88.20, 22: 5.45, 23: 2.60, 24: 46.64, 25: 85.45,
    26: 81.32, 27: 70.77, 28: 73.33
}

# Standard deviations for the measured flows
std_devs = {
    1: 0.25, 2: 0.25, 3: 3, 4: 3, 5: 3, 
    6: 3, 7: 0.25, 8: 3, 9: 0.25, 10: 3,
    11: 1, 12: 3, 13: 3, 14: 3, 15: 3,
    16: 1, 17: 1, 18: 1, 19: 0.25, 20: 1,
    21: 3, 22: 0.25, 23: 0.25, 24: 3, 25: 3,
    26: 3, 27: 3, 28: 3
}

# Objective function
def objective_function(model):
    return sum(((model.F[i] - measured_flows[i]) / std_devs[i])**2 for i in measured_flows)

m.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Constraints based on the provided mass balance equations
def c1_rule(model): 
    return model.F[1] + model.F[2] + model.F[4] == model.F[3]

def c2_rule(model): 
    return model.F[7] + model.F[8] == model.F[5] + model.F[6] + model.F[9]

def c3_rule(model): 
    return model.F[5] == model.F[10] + model.F[1]

def c4_rule(model): 
    return model.F[10] + model.F[11] == model.F[12]

def c5_rule(model): 
    return model.F[3] + model.F[13] == model.F[11] + model.F[14] + model.F[15] + model.F[16] + model.F[17]

def c6_rule(model): 
    return model.F[6] == model.F[2] + model.F[13]

def c7_rule(model): 
    return model.F[14] + model.F[18] == model.F[7] + model.F[19] + model.F[20] + model.F[21]

def c8_rule(model): 
    return model.F[15] + model.F[22] == model.F[18] + model.F[23] + model.F[24]

def c9_rule(model): 
    return model.F[12] + model.F[16] == model.F[22] + model.F[25]

def c10_rule(model): 
    return model.F[23] + model.F[27] + model.F[19] == model.F[26]

def c11_rule(model): 
    return model.F[20] + model.F[26] + model.F[28] == model.F[8]

m.c1 = pyo.Constraint(rule=c1_rule)
m.c2 = pyo.Constraint(rule=c2_rule)
m.c3 = pyo.Constraint(rule=c3_rule)
m.c4 = pyo.Constraint(rule=c4_rule)
m.c5 = pyo.Constraint(rule=c5_rule)
m.c6 = pyo.Constraint(rule=c6_rule)
m.c7 = pyo.Constraint(rule=c7_rule)
m.c8 = pyo.Constraint(rule=c8_rule)
m.c9 = pyo.Constraint(rule=c9_rule)
m.c10 = pyo.Constraint(rule=c10_rule)
m.c11 = pyo.Constraint(rule=c11_rule)

# Solve the model
solver = pyo.SolverFactory('ipopt')
result = solver.solve(m)

# Display the results
for i in range(1, 29):
    print('F[{}] = {:.2f}'.format(i, m.F[i].value))

from scipy.stats import norm

# Reconciled flow values from Part A for the involved nodes
reconciled_flows = {
    7: 2.31, 18: 16.32, 19: 8.0, 20: 10.44, 21: 87.79
}

# Calculate the expected value for F[14] based on the mass balance at node 7
# F[14] + F[18] = F[7] + F[19] + F[20] + F[21]
expected_F14 = reconciled_flows[7] + reconciled_flows[19] + reconciled_flows[20] + reconciled_flows[21] - reconciled_flows[18]
# expected_F14 = 125.21 # Case for Node 5

# Given standard deviation for flow 14
std_dev_F14 = 3

# Z-score for 90% confidence interval (one-tailed test)
z_score_90 = norm.ppf(0.90)

# Calculate the range for F[14]
lower_bound_F14 = expected_F14 - z_score_90 * std_dev_F14
upper_bound_F14 = expected_F14 + z_score_90 * std_dev_F14

# Output the expected value and the range
print(f'Expected value for F[14]: {expected_F14}')
print(f'Lower bound for F[14] at 90% confidence: {lower_bound_F14}')
print(f'Upper bound for F[14] at 90% confidence: {upper_bound_F14}')