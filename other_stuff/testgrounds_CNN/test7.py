import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create some data
data = np.random.rand(10, 10)

# Create labels for each axis point
x_labels = [f'X{i}' for i in range(1, 11)]
y_labels = [f'Y{i}' for i in range(1, 11)]

# Create the heatmap
ax = sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels)

# Set the axis labels and title
ax.set_xlabel('X Axis Label')
ax.set_ylabel('Y Axis Label')
ax.set_title('Heatmap Title')

# Optionally, rotate the x-axis labels for better readability
# ax.set_xticklabels(ax.get_xticklabels(), ha='right')

# Show the plot
# plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
plt.show()