import numpy as np
import random
import time
import math


def run(test_file): 
    # Read file 
    file = open(test_file, 'r')
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
        sol = sol 

        total_cost = 0
        for i in range(m):
            for j in range(i + 1, m):
                idx_i = int(sol[i]) - 1
                idx_j = int(sol[j]) - 1
                total_cost += distances[idx_i][idx_j]
        
        return total_cost / m 

    def repair(sol): 
        # Get unique elements from solution
        unique = list(set(sol))
        
        # If already have m or more unique, just take first m
        if len(unique) >= m:
            return np.array(unique[:m], dtype=int)
        
        # Find missing landmarks
        all_landmarks = set(range(1, c + 1))
        missing = list(all_landmarks - set(unique))
        
        # Combine unique + missing to get exactly m elements
        result = unique + missing[:m - len(unique)]
        
        return np.array(result, dtype=int)


    # One point crossover 
    def create_children(parent1, parent2):
        child = np.zeros(m, dtype=int)

        for i in range(m//2): 
            child[i] = parent1[i]

        for i in range(m//2, m): 
            child[i] = parent2[i]

        child = repair(child)

        return child

    # Mutates a solution by random replacement
    def mutate(sol):
        sol = sol.copy()
        amount = len(sol) // 10
        if amount < 1 : amount = 1
        
        index = random.randint(0, m - 1)
            
        random_replacement = random.randint(1, c)
        sol[index] = random_replacement

        sol = repair(sol)

        return sol



    # Generates a random solution
    def generate_random_solution():
        return np.random.choice(c, m, replace=False) + 1


    # Generates a random population of n solutions
    def generate_population(n):
        population = []
        seen = set()  # ← IMPORTANT: Initialize OUTSIDE the loop
        
        for i in range(n):  # ← Only one loop needed
            while True:
                sol = generate_random_solution()
                sol_tuple = tuple(sorted(sol))  # Position-independent uniqueness
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    population.append(sol)
                    break
        
        return population

    def tournament_selection(population, tournament_size): 
            parents = []
            # Tournament selection
            while len(parents) < 2: 
                tournament = random.sample(population, tournament_size )
                parents.append(max(tournament, key=lambda x: x[1])[0])

            return parents


    def roulette_wheel_selection(population): 
        fitnesses = [being[1] for being in population]
        total_fitness = sum(fitnesses)

        if total_fitness <= 0: 
            return random.choice(population)[0]
        
        pick = random.uniform(0, total_fitness)
        current = 0

        for solution, fitness in population: 
            current += fitness
            if current >= pick: 
                return solution

    def evolutionary_algorithm():
        # track time
        start_time = time.time()

        pop_size = 40

        if c <= 10:  # For small problems
            pop_size = min(pop_size, math.comb(c, m))  # Can't have more unique solutions than possible        
        population = generate_population(pop_size)
        population = [(element, calculate_cost(element)) for element in population]

        # top 50% is parents
        par = len(population) // 2
        print(f'Amount of parents: {par}')

        # top 10% is elite
        elite = len(population) // 10
        if elite < 3 : elite = 3
        print(f'Amount of elite: {elite}')
        
        for generation in range(500):       
            next_gen = []

            population.sort(key=lambda x: -x[1])  # Sort in-place
            for i in range(elite): 
                next_gen.append(population[i][0].copy())  # population[i] is (solution, fitness) tuple
            
            # Generate children through crossover and mutation
            while len(next_gen) < pop_size:
                # Roulette wheel selection
                p1 = roulette_wheel_selection(population)
                p2 = roulette_wheel_selection(population)
                
                child = create_children(p1, p2)
                    
                # 10% mutation rate
                if random.randint(1, 10) == 1:
                    child = mutate(child)
                    
                next_gen.append(child)
            
            # Evaluate new population
            population = [(element, calculate_cost(element)) for element in next_gen]
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time


        # Return best solution
        population.sort(key=lambda x: -x[1])
        print(f'\nFinal best solution: {population[0][0]}')
        print(f'Final best cost: {population[0][1]:.2f}')

        return (population[0][1], population[0][0], 1000, elapsed_time)



    return evolutionary_algorithm()
