import matplotlib.pyplot as plt
import numpy as np

# Test results
actual = np.array([
    -3.190, -1.850, -1.590, -2.680, -3.800,
    -2.369, -2.680, -2.390, 0.320, -1.990
])

predicted = np.array([
    -4.100, -1.656, -1.348, -0.543, -3.156,
    -2.465, -4.794, -2.359, 0.278, -0.620
])

# Create plot
plt.figure(figsize=(10, 7))

plt.scatter(actual, predicted, s=60)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)

plt.xlabel("Actual ESOL Value")
plt.ylabel("Predicted ESOL Value")
plt.title("GNN: Actual vs Predicted ESOL Values")

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
output_file = "gnn_actual_vs_predicted.png"
plt.savefig(output_file, dpi=300)

print()
print("Plot saved successfully!")
print("File name:", output_file)

# Show the graph
plt.show()