import numpy as np

i = 2
j = 3

a = - 0.1 + 0.1 * (i-1)
b = -0.9 + 0.1 * (j-1)

def obj(a, b):
    return 4*a**2 -2.1*a**4 + 1/3 * a**6 + a*b - 4*b**2 + 4*b**4

print(obj(a, b))
print(a)
print(b)