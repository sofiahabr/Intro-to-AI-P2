#  Simulated annealing local search algorithm

import numpy as np
import random
import math


# Read file 
file = open("tourism_500.txt", 'r')
line_list = file.readlines()

# Close the file
file.close()


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
    

"""
Start of helper functions

"""
# Generates a random solution
def generate_random_solution():
    return np.random.choice(c, m, replace=False) + 1

"""
With penalty
"""
# Calculate cost of a solution (sum of pairwise distances)
def calculate_cost(sol):
    sol = sol.astype(int)  

    # If there are any duplicates, the solution's value is 0
    if len(set(sol)) != m: 
        return 0
    
    total_cost = 0
    for i in range(m):
        for j in range(i + 1, m):
            idx_i = sol[i] - 1
            idx_j = sol[j] - 1
            total_cost += distances[idx_i][idx_j]
    
    return total_cost / m  

# Finds a random neighbor of the current solution (similar to mutation) 
def find_random_neighbor(solution):
    solution = solution.copy().astype(int)
    
    # Single-site replacement
    index = random.randint(0, m - 1)
    random_replacement = random.randint(1, c)  
    solution[index] = random_replacement

    return solution, calculate_cost(solution)



"""
The final simulated annealing algorithm

Variables:
    T = 1000
    cooling_rate = 0.95
    iterations = 5000

"""

# Runs simulated annealing algorithm 
def simulated_annealing(): 
    current = generate_random_solution()
    current_cost = calculate_cost(current)

    best = current
    best_cost = current_cost

    # Variables
    T = 1000
    cooling_rate = 0.95

    # Runs the algorithm 5 000 times
    for i in range(5000): 
        # Finds and evaluates a random neighbor of the solution
        neighbor, neighbor_cost = find_random_neighbor(current)
        delta_E = neighbor_cost - current_cost

        if (i + 1)%10 == 0: 
            print(f'Iteration {i + 1}; T = {T:.2f}, best cost: {best_cost:.2f}')

        # If the solution is better, accept it
        if delta_E > 0 : 
            current, current_cost = neighbor, neighbor_cost

            if current_cost > best_cost:
                # setting the new current as best known solution: 
                best, best_cost = current.copy(), current_cost


        else: 
            if T > 1e-10:
                probability = math.exp(delta_E / T)
                if random.random() < probability: 
                    current, current_cost = neighbor, neighbor_cost
        
        T = T * cooling_rate

                

    print()
    print(f'Final solution: {np.sort(best)}')
    print(f'Final cost: {best_cost}')
    return best, best_cost

simulated_annealing()
