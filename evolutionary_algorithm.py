# Evolutionary algorithm 

import numpy as np
import random


# Read file 
file = open("tourism_5.txt", 'r')

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

# A population of solutions
# The best are selected to continue on to the next generation
# The best are selected for breeding, combining so that each solution is combined with another to create 2 children
# There is always a 1 % chance of random mutation 


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

# Combines 2 parents to make 1 child
def create_child(parent1, parent2):
    indices_p1 = np.where(parent1 == 1)[0]
    indices_p2 = np.where(parent2 == 1)[0]

    combined_indices = []
    child = np.zeros(c)

    for i in range(m//2): 
        index = indices_p1[i]
        combined_indices.append(index)
        child[index] = 1

    for i in range(1, m + 1): 
        if indices_p2[-i] not in combined_indices and len(combined_indices) < m: 
            index = indices_p2[-i]
            combined_indices.append(index)
            child[index] = 1

    return child

def create_children(p1, p2): 
    return create_child(p1, p2), create_child(p2, p1)

# Mutates a solution using swap / random replacement
# Since we have t have the same amount of 1 in each solution we have to tick one down as we tick one up
# This results in swap mutation
def mutate(sol):
    indices_1 = np.where(sol == 1)[0]
    indices_0 = np.where(sol == 0)[0]

    # choses the indexes of the bits were swapping
    index0 = indices_0[random.randint(0, len(indices_0) - 1)]
    index1 = indices_1[random.randint(0, len(indices_1) - 1)]

    # swaps the indexes
    sol[index1] = 0
    sol[index0] = 1

    return sol 


# Generates a random solution
def generate_random_solution() : 
    solution = np.zeros(c)

    while True:
        i = random.randint(0,c - 1)
        if solution.sum() < m: 
            solution[i] = 1 - solution[i]
        else : 
            break

    return solution


# Generates a random popultaion of n solutions 
def generate_population(n) : 
    list = []

    for i in range(n): 
        sol = generate_random_solution()
        while any(np.array_equal(sol, s) for s in list): 
            print('Solution already exists in list')
            sol = generate_random_solution()

        list.append(sol)

    return list


p1 = generate_random_solution()

print(p1)
for i in range(50):
    print(mutate(p1))