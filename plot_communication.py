import matplotlib.pyplot as plt
import numpy as np

# Number of agents
agents = [5, 10, 20]

# Algorithms and their corresponding values for fully connected and ring topologies
algorithms = {
    '(DIMAT - WA)': {
        'Fully Connected': [0.5 * (n - 1) for n in agents],
        'Ring': [0.5 * 2 for _ in agents],
    },
    '(SGP - CDSGD)': {
        'Fully Connected': [100 * (n - 1) for n in agents],
        'Ring': [100 * 2 for _ in agents],
    },
    # 'CGA': {
    #     'Fully Connected': [200 * (n - 1) for n in agents],
    #     'Ring': [200 * 2 for _ in agents],
    # },
}

# Organize values for plotting
values = []
labels = []

# Collect values and labels for each combination of algorithm and topology
for top in ['Ring', 'Fully Connected']:
    for algo, tops in algorithms.items():
        labels.append(f"{algo} - {top}")
        values.append(tops[top])

# Convert to numpy array for reshaping
values = np.array(values)

# Reshape values for plotting (4 bars for each agent)
values = values.reshape(-1, len(agents))
print(values)

# Create separate bar plots for each topology
bar_width = 0.15
index = np.arange(len(agents))

# Define the provided color map
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

fig, ax = plt.subplots(figsize=(8, 5), dpi=1200)  # Set the figure size and resolution with increased height

# Plotting bars for each combination of agents, algorithm, and topology with specific colors
for i in range(len(labels)):
    ax.bar(index + i * bar_width, values[i], bar_width, label=labels[i], color=colors[i])

# Configure plot
ax.set_xlabel('Number of Models', fontsize=15)
ax.set_ylabel('Communication Rounds per Epoch', fontsize=15)
# ax.set_title('Comparison of Algorithms for Fully Connected and Ring Topologies')
ax.set_xticks(index + bar_width * len(labels) / 2)
ax.set_xticklabels(agents)

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Position legend below the figure
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2, fontsize=15)

# Show the plot
plt.savefig('results/figures/communication_rounds2.png', bbox_inches='tight')  # Save the plot with tight bounding box
plt.show()
