from sympy import symbols, hessian, exp, log, simplify

# Define the variables
x1, x2, x3 = symbols('x1 x2 x3', real=True)

# Define the function f
# f = x3 * log(exp(x1/x3) + exp(x2/x3)) + (x3 - 2)**2 + exp(1/(x1 + x2))
f = (x3 - 2)**2

# Compute the Hessian matrix of f
H = hessian(f, (x1, x2, x3))
H = simplify(H)  # Simplify the Hessian matrix for better readability

# Display the Hessian Matrix
print(H)