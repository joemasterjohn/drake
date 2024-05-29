import matplotlib.pyplot as plt

# Read the file and extract cycle numbers
filename = 'output_no_divergence.txt'
cycle_numbers = []

with open(filename, 'r') as file:
    for line in file:
        if 'cycles [k]:' in line:
            cycle_number = int(line.split(': ')[1])
            cycle_numbers.append(cycle_number)

# Plot the histogram
plt.hist(cycle_numbers, bins=50, edgecolor='black')
plt.title('Distribution of Cycle Numbers, Not Radomized, 66 DOF Case, 2000 Envs')
plt.xlabel('Number of Cycles (in 1k)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
