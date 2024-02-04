# Define the objective function
def objective(x1, x2):
    return -2 * x1 - 3 * x2

# Create a grid of points
x1 = np.linspace(0, 4, 800)
x2 = np.linspace(0, 3, 600)
X1, X2 = np.meshgrid(x1, x2)

# Compute the value of the function at each point
Z = objective(X1, X2)

# Create a mask for the feasible region
feasible_region = np.logical_and(X1 + X2 <= 4, X1 + 2*X2 <= 6)

# Mask the objective function outside the feasible region
Z_masked = np.where(feasible_region, Z, np.nan)

# Plot the contours of the function
fig, ax = plt.subplots(figsize=(8, 6))
contour_plot = ax.contourf(X1, X2, Z_masked, 50)
plt.colorbar(contour_plot)

# Mark the optimal point
optimal_x1 = 2
optimal_x2 = 2
optimal_value = objective(optimal_x1, optimal_x2)
plt.plot(optimal_x1, optimal_x2, 'ro')  # ro represents red color and dot marker
plt.annotate('Optimal: Z(%.1f, %.1f) = %.1f' % (optimal_x1, optimal_x2, optimal_value), 
             (optimal_x1, optimal_x2), 
             textcoords="offset points", 
             xytext=(-100,-10))

# Set labels and title
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Contour Plot of Objective and Feasible Region')

# Add label for the objective function
ax.text(0.2, 1.5, r'$Z = -2x_1 - 3x_2$', fontsize=12, color='black')

# Show the plot
plt.grid()
plt.show()