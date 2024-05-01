# Job Shop Scheduling Problem (JSSP) - Infeasible Model

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

# Assuming Task dictionary is already defined
# Creating a cyclic dependency that is logically impossible
Task = {
    ('A', 1): {'dur': 5, 'prec': ('C', 2)},  # A1 now depends on C2
    ('A', 3): {'dur': 3, 'prec': ('A', 1)},
    ('B', 2): {'dur': 3, 'prec': None},
    ('B', 3): {'dur': 2, 'prec': ('B', 2)},
    ('C', 1): {'dur': 2, 'prec': None},
    ('C', 2): {'dur': 4, 'prec': ('A', 3)}  # C2 depends on A3 creating a cycle with A1
}

model = pyo.ConcreteModel()

model.TASKS = pyo.Set(initialize=Task.keys(), dimen=2, doc="Tasks")
model.JOBS = pyo.Set(initialize=list(set(j for (j, m) in model.TASKS)), doc="Jobs")
model.STAGES = pyo.Set(initialize=list(set(m for (j, m) in model.TASKS)), doc="Stages")

# Parameters
model.dur = pyo.Param(model.TASKS, initialize=lambda model, j, m: Task[(j, m)]['dur'], doc="Duration of each task", mutable=True)

# Variables
ub = sum(Task[jm]['dur'] for jm in Task)
model.makespan = pyo.Var(bounds=(0, ub)) 
model.start = pyo.Var(model.TASKS, bounds=(0, ub))

# Binary variables for task ordering
bigM = ub  # Setting bigM to the upper bound of makespan
model.y = pyo.Var(model.TASKS * model.TASKS, within=pyo.Binary, doc="Task ordering")

# Objective
model.objective = pyo.Objective(expr=model.makespan, sense=pyo.minimize, doc="Minimize makespan")

# Constraints
model.finish = pyo.Constraint(model.TASKS, rule=lambda model, j, m:
                              model.start[j, m] + model.dur[j, m] <= model.makespan, doc="Task finish time")

model.preceding = pyo.ConstraintList(doc="Preceding tasks")
for (j, m) in Task:
    if Task[(j, m)]['prec']:
        k, n = Task[(j, m)]['prec']
        model.preceding.add(model.start[k, n] + model.dur[k, n] <= model.start[j, m])

# Disjunctions for task ordering
model.disjunctions = pyo.ConstraintList(doc="Disjunctions")
for (j, m) in model.TASKS:
    for (k, n) in model.TASKS:
        if m == n and (j, m) != (k, n):  # only if same machine and different tasks
            model.disjunctions.add(model.start[j, m] + model.dur[j, m] <= model.start[k, n] + bigM * (1 - model.y[(j, m), (k, n)]))
            model.disjunctions.add(model.start[k, n] + model.dur[k, n] <= model.start[j, m] + bigM * model.y[(j, m), (k, n)])

