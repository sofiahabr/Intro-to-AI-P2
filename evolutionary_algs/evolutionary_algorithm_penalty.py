import numpy as np
import random


# Read file 
file = open("tourism_500.txt", 'r')
line_list = file.readlines()

# Close the file
file.close()

# C is the amount of total landmarks, and m is the amount to be in the solution
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


# Calculate cost of a solution
def calculate_cost(sol):
    sol = sol.astype(int) 

    # For penalty 
    if len(set(sol)) != m: return 0
    
    total_cost = 0
    for i in range(m):
        for j in range(i + 1, m):
            idx_i = sol[i] - 1
            idx_j = sol[j] - 1
            total_cost += distances[idx_i][idx_j]
    
    return total_cost / m  # Return avg distance between landmarks

"""
# Combines 2 parents to make 1 child using crossover
def create_children(parent1, parent2):
    parent1 = parent1.astype(int)
    parent2 = parent2.astype(int)
    
    if np.array_equal(parent1, parent2):
        return parent1.copy()
    
    remaining = np.array([x for x in parent2 if x not in parent1])
    parents = [*parent1, *remaining]

    child = np.zeros(m, dtype=int)
    
    for i in range(m // 2):
        child[i] = parents[i]
    
    for i in range(m // 2, m):
        child[i] = parents[- 1 + (m//2 - i)]
    
    return child

"""

# One point crossover 
def create_children(parent1, parent2):
    parent1 = np.sort(parent1)
    parent2 = np.sort(parent2)

    child = np.zeros(m, dtype=int)

    for i in range(m//2): 
        child[i] = parent1[i]

    for i in range(m//2, m): 
        child[i] = parent2[i]

    return child
"""

# Uniform crossover 
def create_children(parent1, parent2):
    parent1 = np.sort(parent1)
    parent2 = np.sort(parent2)
    
    child = np.zeros(m, dtype=int)

    for i in range(m):
        if random.randint(0,1) == 0: 
            child[i] = parent1[i] 

        else: 
            child[i] = parent2[i]

    return child
"""

# Mutates a solution by swap mutation / random replacement
def mutate(sol):
    sol = sol.astype(int)
    index = random.randint(0, m - 1)
    
    random_replacement = random.randint(1, c)
    sol[index] = random_replacement
    return sol

"""
# Mutates a solution by two-site swap mutation
def mutate(sol):
    sol = sol.astype(int)
    index = random.randint(0, m - 1)
    index2 = random.randint(0, m - 1)
    
    random_replacement = random.randint(1, c)
    random_replacement_2 = random.randint(1, c)

    sol[index] = random_replacement
    sol[index2] = random_replacement_2
    return sol
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


# Generates a random population of n solutions
def generate_population(n):
    population = []
    
    for i in range(n):
        sol = generate_random_solution()
        while any(np.array_equal(sol, s) for s in population):
            sol = generate_random_solution()
        population.append(sol)
    
    return population


def evolutionary_algorithm():
    pop_size = 40
    population = generate_population(pop_size)
    population = [(element, calculate_cost(element)) for element in population]

    # top 50% is parents
    top = len(population) // 2
    print(f'Amount of parents: {top}')

    # top 10% is elite
    elite = len(population) // 10
    if elite < 3 : elite = 3
    print(f'Amount of elite: {elite}')
    
    for generation in range(500):
        # Sort by cost
        population.sort(key=lambda x: -x[1])
        
        if (generation + 1)%10 == 0: 
            avg_fitness = sum([element[1] for element in population]) / len(population)
            print(f'Generation {generation + 1}: Avg cost = {avg_fitness:.2f}, Best = {population[0][1]:.2f}')
        
        # Keep top 50% as parents for the next gen
        population = population[:top]
        
        next_gen = []
        
        # Elitism: keep the best 10%
        for i in range(elite): 
            next_gen.append(population[i][0].copy())
        
        # Generate children through crossover and mutation
        while len(next_gen) < pop_size:
            p1_index = random.randint(0, len(population) - 1)
            p2_index = random.randint(0, len(population) - 1)
            
            if p1_index != p2_index:
                child = create_children(population[p1_index][0], population[p2_index][0])
                
                # 10% mutation rate
                if random.randint(1, 10) == 1:
                    child = mutate(child)
                
                next_gen.append(child)
        
        # Evaluate new population
        population = [(element, calculate_cost(element)) for element in next_gen]
    
    # Return best solution
    population.sort(key=lambda x: -x[1])
    print(f'\nFinal best solution: {np.sort(population[0][0])}')
    print(f'Final best cost: {population[0][1]:.2f}')
    return population[0]


evolutionary_algorithm()