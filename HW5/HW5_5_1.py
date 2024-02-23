import numpy as np
import matplotlib.pyplot as plt

# Intersection points
intersection_points = [(1/2 - np.sqrt(3)/2, -np.sqrt(3)/2 - 1/2), (1/2 + np.sqrt(3)/2, np.sqrt(3)/2 - 1/2)]

# x1 and x2 non-negative real
x1 = np.linspace(0, 2, 100)
x2 = np.linspace(0, 2, 100)

# Create meshgrid
X1, X2 = np.meshgrid(x1, x2)

# Constraints x1^2 + x2^2 <= 2 and x1 - x2 <= 1
F = X1**2 + X2**2 - 2
G = X1 - X2 - 1

# Plot the constraints
plt.figure(figsize=(6, 6))
plt.contour(X1, X2, F, [0], colors='r')
plt.contour(X1, X2, G, [0], colors='b')

# Correctly fill the feasible region
x_fill = np.linspace(intersection_points[0][0], intersection_points[1][0], 100)
y_fill_lower = np.maximum(0, x_fill - 1)
y_fill_upper = np.sqrt(2 - x_fill**2)

plt.fill_between(x_fill, y_fill_lower, y_fill_upper, color='grey', alpha=0.5)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.title('Feasible Region')
plt.grid(True)
plt.show()


