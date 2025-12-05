# Simultaed annealing

"""
Simulates shaking a ball out of smaller groves. 
Starts with large shakes, then reduces the intencity untill it cant dislodge any balls anymore. 

The overall structure is similar to hill climbing. 
Instead of picking the best move, it picks a random move, and if the move improves the situation it is always accepted. 
Otherwise the algoritm accepts the move with a probability of less than 1. 
The probability decreases exponentially with the badness of the move, the amount $\Delta E$  by which the equation is worsened. 
The probability decreases as the tempature T goes down. 
Bad moves become more unlikely as the T decreases, until it reaches 0.
"""
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
    solution = np.zeros(m, dtype=int)
    
    for i in range(m):
        landmark = random.randint(1, c)
        while landmark in solution:
            landmark = random.randint(1, c)
        solution[i] = landmark
    
    return solution

"""
With prevention / Constraint aware design
"""
# Calculate cost of a solution (sum of pairwise distances)
def calculate_cost(sol):
    sol = sol.astype(int)  
    total_cost = 0
    for i in range(m):
        for j in range(i + 1, m):
            idx_i = sol[i] - 1
            idx_j = sol[j] - 1
            total_cost += distances[idx_i][idx_j]
    
    return total_cost / m   

# Finds a random neighbor of the current solution (similar to mutation) 
def find_random_neighbor(solution):
    solution = solution.astype(int)
    index = random.randint(0, m - 1)
    
    random_replacement = random.randint(1, c)
    while random_replacement in solution:
        random_replacement = random.randint(1, c)
    
    solution[index] = random_replacement

    return solution, calculate_cost(solution)

"""
The final algorithm that iterates through simulated annealing
"""

# Runs simulated annealing algorithm 
def simulated_annealing(): 
    current = generate_random_solution()
    current_cost = calculate_cost(current)

    best = current
    best_cost = current_cost

    # Variables
    T = 1000
    cooling_rate = 0.955

    # Runs the algorithm 500 times
    for i in range(500): 
        # Finds and evaluates a random neighbor of the solution
        neighbor, neighbor_cost = find_random_neighbor(current)
        delta_E = neighbor_cost - current_cost

        if (i + 1)%10 == 0: 
            print(f'Iteration {i + 1}; T = {T}, best cost: {best_cost}')

        # If the solution is better, accept it
        if delta_E > 0 : 
            current, current_cost = neighbor, neighbor_cost

            # setting the new current as best known solution: 
            best, best_cost = current, current_cost
        
        else: 
            probability = math.exp(delta_E / T)
            if random.random() < probability: 
                current, current_cost = neighbor, neighbor_cost
                
                """
        if T < 0.001: 
            print(f'Iteration {i + 1}; T = {T}, best cost: {best_cost}')
            break

                """
            
        T = T * cooling_rate

    print()
    print(f'Final solution: {np.sort(best)}')
    print(f'Final cost: {best_cost}')
    return best, best_cost

simulated_annealing()
