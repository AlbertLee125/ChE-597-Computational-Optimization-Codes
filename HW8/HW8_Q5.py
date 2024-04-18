import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Squares lengths
square_lengths = [12, 10, 9, 6, 3, 2]

# Start with an empty rectangle
rectangle_width = 0
rectangle_height = 0

# Track the positions of squares
positions = []

# Place each square
current_x = 0
current_y = 0
row_height = 0

for length in square_lengths:
    # Check if adding the square exceeds the maximum allowed dimension
    if current_x + length > 25 or current_y + row_height > 25:
        print(f"Cannot place square with length {length} without exceeding 25 units.")
        break  # Exit the loop if the condition is met

    # Check if the square fits in the current row
    if current_x + length > rectangle_width:
        # Start a new row
        current_x = 0
        current_y += row_height
        row_height = length

    # Update rectangle dimensions
    rectangle_width = max(rectangle_width, current_x + length)
    rectangle_height = max(rectangle_height, current_y + length)

    # Save the position and move to the next position
    positions.append((current_x, current_y))
    current_x += length

# Plotting the arrangement up to the point where a square could not be placed
fig, ax = plt.subplots()
rectangle = patches.Rectangle((0, 0), rectangle_width, rectangle_height, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rectangle)

for i, (x, y) in enumerate(positions):
    square = patches.Rectangle((x, y), square_lengths[i], square_lengths[i], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(square)
    plt.text(x + square_lengths[i]/2, y + square_lengths[i]/2, str(square_lengths[i]), ha='center', va='center')

ax.set_xlim(0, rectangle_width + 1)
ax.set_ylim(0, rectangle_height + 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

print(f"Rectangle dimensions so far: {rectangle_width} x {rectangle_height}")

