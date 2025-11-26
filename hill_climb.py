import numpy as np
import random

#  Hill climbing local search algorithm

# Read file 
file = open("tourism_100.txt", 'r')

line_list = file.readlines()


# C is the ammount of total landmarks, and m is the amount to be in the solution
c, m = line_list[0].split()

c = int(c)
m = int(m)

line_list.pop(0)
lines = []

for i in range(len(line_list)):
    lines.append(line_list[i].split())


# Create a matrix to keep track of distances
distances = np.zeros((c, c))

for line in lines:
    i = int(line[0][1:]) - 1  
    j = int(line[1][1:]) - 1  
    dist = float(line[2])
    
    distances[i][j] = dist
    distances[j][i] = dist


# Generate a random beginning solution
# A soultion is represented by binary list, where 1 means the landmark is included and 0 means it isnt 
def find_random_start() : 
    solution = np.zeros(c)

    while True:
        i = random.randint(0,c - 1)
        if solution.sum() < m: 
            solution[i] = 1 - solution[i]
        else : 
            break

    return solution

"""
# Calculate cost of a solution 
def calculate_cost(sol):
    indicies = np.where(sol == 1)[0]

    names = []
    for i in range(len(indicies)): 
        names.append('e' + str(indicies[i] + 1))

    sum = 0; 
    for i in range(len(lines)): 
        if lines[i][0] in names and lines[i][1] in names: 
            sum += float(lines[i][2])

    return sum / m

"""

# Calculate cost of a solution 
def calculate_cost(sol):
    indices = np.where(sol == 1)[0]

    sum = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i = indices[i]
            idx_j = indices[j]
            sum += distances[idx_i][idx_j]

    return sum / m


# Find neighbors
# To find neighbors we have to flip the bit of one of the 1 bits, 
# and flip the bit of one of the 0 bits.  
def find_neighbors(sol): 
    indicies_1 = np.where(sol == 1)[0]
    indicies_0 = np.where(sol == 0)[0]

    neighbors = []

    for i in indicies_1:
        neighbor = sol.copy()     
        neighbor[i] = 0; 
        for j in indicies_0:
            neighbor_i = neighbor.copy()
            neighbor_i[j] = 1
            neighbors.append(neighbor_i)

    return neighbors

# Run the hill climbing search algorithm 
def hill_climbing(): 
    solution = find_random_start()
    # print(f'start solution: {solution}')

    it = 0
    while True : 
        it += 1
        print(f'iteration {it}')

        # calculate cost of current solution
        cost = calculate_cost(solution)

        # find neighbors
        neighbors = find_neighbors(solution)

        # find the cost of each of the neighbors
        cost_neighbors = []

        for neighbor in neighbors:
            cost_neighbors.append(calculate_cost(neighbor))


        index = np.argmax(cost_neighbors)
        if cost_neighbors[index] > cost: 
            solution = neighbors[index]
        else: 
            break
    return solution

solution = hill_climbing()
print(f'Final solution: {solution}')
print(f'Final cost: {calculate_cost(solution)}')
# Close the file
file.close()