import numpy as np
import random
import math
import time


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
        seen = set()
        
        for i in range(n):
            while True:
                sol = generate_random_solution()
                sol_tuple = tuple(sorted(sol))
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    population.append(sol)
                    break
        
        return population

    def tournament_selection(population, tournament_size): 
        parents = []
        while len(parents) < 2: 
            tournament = random.sample(population, tournament_size)
            parents.append(max(tournament, key=lambda x: x[1])[0])

        return parents

    # ============================================================================
    # SIMULATED ANNEALING REFINEMENT (embedded in EA)
    # ============================================================================
    
    def refine_with_sa(solution, num_iterations=100):
        """Refine a solution using Simulated Annealing"""
        current = solution.copy().astype(int)
        current_cost = calculate_cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        T = 100.0  # Initial temperature for refinement
        cooling_rate = 0.95
        
        for iteration in range(num_iterations):
            # Find random neighbor
            neighbor = current.copy()
            index = random.randint(0, m - 1)
            random_replacement = random.randint(1, c)
            neighbor[index] = random_replacement
            neighbor = repair(neighbor)
            
            neighbor_cost = calculate_cost(neighbor)
            delta_E = neighbor_cost - current_cost
            
            # Accept or reject
            if delta_E > 0:
                current = neighbor.copy()
                current_cost = neighbor_cost
                
                if current_cost > best_cost:
                    best = current.copy()
                    best_cost = current_cost
            else:
                if T > 1e-10:
                    probability = math.exp(delta_E / T)
                    if random.random() < probability:
                        current = neighbor.copy()
                        current_cost = neighbor_cost
            
            T = T * cooling_rate
        
        return best, best_cost

    # ============================================================================
    # HYBRID EVOLUTIONARY ALGORITHM WITH EMBEDDED SA
    # ============================================================================

    def hybrid_embedded():
        # track time
        start_time = time.time()

        pop_size = 40

        if c <= 10:  # For small problems
            pop_size = min(pop_size, math.comb(c, m))

        population = generate_population(pop_size)
        population = [(element, calculate_cost(element)) for element in population]

        # top 10% is elite
        elite = len(population) // 10
        if elite < 3 : elite = 3
        print(f'Amount of elite: {elite}')
        print(f'Population size: {pop_size}')
        
        for generation in range(500):       
            next_gen = []

            if (generation + 1) % 100 == 0 or generation == 0: 
                best_cost = population[0][1]
                print(f'Generation {generation + 1}; Best cost = {best_cost:.6f}')


            population.sort(key=lambda x: -x[1])
            
            # Keep elite and refine them with SA
            for i in range(elite): 
                elite_solution = population[i][0].copy()
                
                # Refine elite with SA (50 iterations each)
                refined_solution, refined_cost = refine_with_sa(elite_solution, num_iterations=50)
                
                next_gen.append(refined_solution)
            
            # Generate children through crossover and mutation
            while len(next_gen) < pop_size:
                # Tournament selection
                p1, p2 = tournament_selection(population, 10)
            
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



    return hybrid_embedded()
