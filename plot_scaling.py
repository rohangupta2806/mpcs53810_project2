import matplotlib.pyplot as plt

x_values = [10, 31, 100, 310, 1000, 3100]
y_values = [2.59, 4.88, 10.26, 23.95, 63.73, 171.52]

# Create a figure and axis
fig, ax = plt.subplots()
# Plot the data
ax.plot(x_values, y_values, marker='o')
# Set the title and labels
ax.set_title('Scaling of Number of Proposals with respect to n')
ax.set_xlabel('n')
ax.set_ylabel('Number of Proposals')
# Set the x-axis to logarithmic scale
ax.set_xscale('log')
plt.xticks(x_values)
# Show the plot
plt.show()

x_values = [10, 31, 100, 310, 1000, 3100]
y_values = [2.59, 4.88, 10.26, 23.95, 63.73, 171.52]

# Create a figure and axis
fig, ax = plt.subplots()
# Plot the data
ax.plot(x_values, y_values, marker='o')
# Set the title and labels
ax.set_title('Scaling of average doctor ranking with respect to n')
ax.set_xlabel('n')
ax.set_ylabel('Average doctor ranking')
# Set the x-axis to logarithmic scale
ax.set_xscale('log')
plt.xticks(x_values)
# Show the plot
plt.show()

x_values = [10, 31, 100, 310, 1000, 3100]
y_values = [3.70, 7.44, 10.26, 36.78, 87.26, 218.74]

# Create a figure and axis
fig, ax = plt.subplots()
# Plot the data
ax.plot(x_values, y_values, marker='o')
# Set the title and labels
ax.set_title('Scaling of average hospital ranking with respect to n')
ax.set_xlabel('n')
ax.set_ylabel('Average hospital ranking')
# Set the x-axis to logarithmic scale
ax.set_xscale('log')
plt.xticks(x_values)
# Show the plot
plt.show()


