from scipy.optimize import linprog

# Coefficients of the objective function (we can set this to zero since we're only checking for feasibility)
c = [0, 0, 0]

# Coefficients of the inequality constraints (left-hand side)
A = [[2, 3, 4],  # 2x1 + 3x2 + 4x3 <= 5
     [1, 0, 1],   # x1 + x3 <= 1
     [0, 1, 1]]   # x2 + x3 <= 1

# Constants of the inequality constraints (right-hand side)
b = [5, 1, 1]

# Bounds for each variable
x0_bounds = (0, 1)
x1_bounds = (0, 1)
x2_bounds = (0, 1)

# Solve the linear program
result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds, x2_bounds], method='highs')

print(result)

