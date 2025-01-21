import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the visualization
sns.set(style='whitegrid')

# Load the 'tips' dataset
tips = sns.load_dataset('tips')
print(tips.to_string())



# Create a violin plot with 'day' on the x-axis and 'tip' on the y-axis
sns.violinplot(x='day', y='tip', hue='sex',  split=True, gap=.1, inner='quart', density_norm='count', bw_adjust=1, data=tips)

# Display the plot
plt.show()