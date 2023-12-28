import matplotlib.pyplot as plt

# Read the data from the file
with open('omegas.txt', 'r') as file:
    data = [float(line.strip()) for line in file.readlines()]

# Remove 0 values from the data
data = [value for value in data if value != 0]
print(data)
# Create a bar chart
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)  # Create a subplot for the bar chart
plt.bar(range(len(data)), data)
plt.title('Bar Chart of Omegas')
plt.xlabel('Index')
plt.ylabel('Value')

# Create a box plot with more inclusive whiskers
plt.subplot(2, 1, 2)  # Create a subplot for the box plot
plt.boxplot(data)
plt.title('Box Plot of Omegas')
plt.ylabel('Value')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
