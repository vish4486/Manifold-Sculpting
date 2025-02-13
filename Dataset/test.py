import numpy as np
import matplotlib.pyplot as plt

# Function to generate Swiss Roll data
def generate_swiss_roll(n_samples=1000):
    t = np.linspace(2, 10, n_samples)  # Spiral parameter
    x = t * np.sin(t)
    z = t * np.cos(t)
    y = np.random.uniform(-6, 6, n_samples)  # Random height variation

    data = np.vstack((x, y, z)).T
    color = t  # Used for coloring points

    return data, color

# Generate Swiss Roll Data
swiss_roll_data, swiss_roll_color = generate_swiss_roll(n_samples=1000)

# Plot the Swiss Roll
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(swiss_roll_data[:, 0], swiss_roll_data[:, 1], swiss_roll_data[:, 2], 
                      c=swiss_roll_color, cmap='plasma', alpha=0.8)
ax.set_title("Swiss Roll Dataset")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.colorbar(scatter, label="Color Gradient")

# Save the plot
plot_path = "Plots/swiss_roll_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

# Display the plot
plt.show()


