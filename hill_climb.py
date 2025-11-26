import numpy as np
import random

#  Hill climbing local search algorithm

# Read file 
file = open("tourism_5.txt", 'r')

lines = file.readlines()

# C is the ammount of landmarks, and m is the amount to be in the solution
c, m = lines[0].split()
print(f'c: {c}, m: {m}')

c = int(c)
m = int(m)

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

print(find_random_start())

# Calculate cost of a solution 
def claculate_cost(sol):
    return 

# Find neighbors
def find_neighbors(sol): 
    return

# Run the hill climbing search algorithm 

# Close the file
file.close()