import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create example processed_data
np.random.seed(10)
time = np.tile(np.arange(10), 3)
value = np.random.randn(30).cumsum()
category = np.repeat(['A', 'B', 'C'], 10)

data = pd.DataFrame({'Time': time, 'Value': value, 'Category': category})

# Display the processed_data
print(data)

# Plot multiple lines
sns.lineplot(x='Time', y='Value', hue='Category', data=data)

# Set the axis labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Multiple Lines Plot')

# Show the plot
plt.show()