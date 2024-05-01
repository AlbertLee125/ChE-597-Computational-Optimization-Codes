# Job Shop Scheduling Problem (JSSP)

# Three jobs (A,B,C) must be executed sequentially in three steps, but not all jobs require all the stages. 
# The objective is to obtain the sequence of tasks which minimizes the completion time. 
# Once a job has started it cannot be interrupted. 
# The objective is to obtain the sequence of task, which minimizes the completion time.
# The original model is a linear Generalized Disjunctive Programming (GDP) model and the given code is reformulated as a Mixed Integer Linear Programming (MILP) model.

# References: Yee, T. F., & Grossmann, I. E. (1990). Simultaneous optimization models for heat integration—II. Heat exchanger network synthesis. Computers & Chemical Engineering, 14(10), 1165-1184.

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

# Data
JOBS = ['A', 'B', 'C']
STAGES = [1, 2, 3]
PROCESSING_TIME = {
    ('A', 1): 5, ('A', 3): 3,
    ('B', 2): 3, ('B', 3): 2,
    ('C', 1): 2, ('C', 2): 4
}

# Sets
model.JOBS = pyo.Set(initialize=JOBS, ordered=True)
model.STAGES = pyo.Set(initialize=STAGES, ordered=True)
model.LESS = pyo.Set(within=model.JOBS * model.JOBS, initialize=[(i, j) for i in JOBS for j in JOBS if JOBS.index(i) < JOBS.index(j)])

# Parameters
model.p = pyo.Param(model.JOBS, model.STAGES, initialize=PROCESSING_TIME, default=0, mutable=True, doc="Processing time for each job and stage")

# Calculated parameters
def calc_c(model, j, s):
    """Calculate the cumulative processing time for job j up to stage s"""
    return sum(model.p[j, ss] for ss in model.STAGES if ss <= s)
model.c = pyo.Param(model.JOBS, model.STAGES, initialize=calc_c, doc="Cumulative processing time for each job and stage")

def calc_w(model, j, jj):
    """Calculate the waiting time between jobs j and jj"""
    return max(model.c[j, s] - (model.c[jj, s-1] if s > 1 else 0) for s in model.STAGES)
model.w = pyo.Param(model.LESS, initialize=calc_w, doc="Waiting time between jobs j and jj")

def total_processing_time(model, j):
    """Calculate the total processing time for job j"""
    return sum(model.p[j, s] for s in model.STAGES)
model.pt = pyo.Param(model.JOBS, initialize=total_processing_time, doc="Total processing time for each job")

# Variables
model.t = pyo.Var(domain=pyo.NonNegativeReals, doc="Make span")
model.x = pyo.Var(model.JOBS, domain=pyo.NonNegativeReals, doc="Start time for each job")
model.pr = pyo.Var(model.LESS, domain=pyo.Binary, doc="Precedence relationship between jobs j and jj")

# Constraints
def completion_time_constraint(model, j):
    """Completion time constraint for each job"""
    return model.t >= model.x[j] + model.pt[j]
model.comp = pyo.Constraint(model.JOBS, rule=completion_time_constraint, doc="Completion time constraint for each job")

def sequencing_constraint(model, j, jj):
    """Sequencing constraint between jobs j and jj"""
    if j < jj:
        return model.x[j] + model.w[j, jj] <= model.x[jj]
    else:
        return model.x[jj] + model.w[jj, j] <= model.x[j]
model.seq = pyo.Constraint(model.LESS, rule=sequencing_constraint, doc="Sequencing constraint between jobs j and jj")

# Objective
model.objective = pyo.Objective(expr=model.t, sense=pyo.minimize)

# Solver setup and solve
solver = pyo.SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# Display results
print("Objective Value (Make Span):", pyo.value(model.t))
for j in model.JOBS:
    print(f"Start time for Job {j}: {pyo.value(model.x[j])}")
