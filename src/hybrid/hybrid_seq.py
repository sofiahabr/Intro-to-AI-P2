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


    # ============================================================================
    # SHARED FUNCTIONS
    # ============================================================================

    def calculate_cost(sol):
        """Calculate cost (average distance)"""
        total_cost = 0
        for i in range(m):
            for j in range(i + 1, m):
                idx_i = int(sol[i]) - 1
                idx_j = int(sol[j]) - 1
                total_cost += distances[idx_i][idx_j]
        
        return total_cost / m 

    def repair(sol): 
        """Repair solution to have exactly m unique elements"""
        unique = list(set(sol))
        
        if len(unique) >= m:
            return np.array(unique[:m], dtype=int)
        
        all_landmarks = set(range(1, c + 1))
        missing = list(all_landmarks - set(unique))
        result = unique + missing[:m - len(unique)]
        
        return np.array(result, dtype=int)

    def generate_random_solution():
        """Generate random solution"""
        return np.random.choice(c, m, replace=False) + 1

    # ============================================================================
    # SIMULATED ANNEALING
    # ============================================================================

    def simulated_annealing(): 
        """Run Simulated Annealing"""
        start_time = time.time()

        current = generate_random_solution()
        current_cost = calculate_cost(current)

        best = current
        best_cost = current_cost

        T = 1000
        cooling_rate = 0.95

        for i in range(5000): 
            neighbor = current.copy().astype(int)
            index = random.randint(0, m - 1)
            random_replacement = random.randint(1, c)
            neighbor[index] = random_replacement
            neighbor = repair(neighbor)
            
            neighbor_cost = calculate_cost(neighbor)
            delta_E = neighbor_cost - current_cost

            if delta_E > 0: 
                current, current_cost = neighbor, neighbor_cost

                if current_cost > best_cost:
                    best, best_cost = current.copy(), current_cost

            else: 
                if T > 1e-10:
                    probability = math.exp(delta_E / T)
                    if random.random() < probability: 
                        current, current_cost = neighbor, neighbor_cost
            
            T = T * cooling_rate

        elapsed_time = time.time() - start_time
        return (best_cost, best, 5000, elapsed_time)

    # ============================================================================
    # EVOLUTIONARY ALGORITHM
    # ============================================================================

    def create_children(parent1, parent2):
        """One point crossover"""
        child = np.zeros(m, dtype=int)

        for i in range(m//2): 
            child[i] = parent1[i]

        for i in range(m//2, m): 
            child[i] = parent2[i]

        child = repair(child)
        return child

    def mutate(sol):
        """Mutate solution"""
        sol = sol.copy()
        index = random.randint(0, m - 1)
        random_replacement = random.randint(1, c)
        sol[index] = random_replacement
        sol = repair(sol)
        return sol

    def generate_population(n):
        """Generate random population of n solutions"""
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
        """Tournament selection"""
        parents = []
        while len(parents) < 2: 
            tournament = random.sample(population, tournament_size)
            parents.append(max(tournament, key=lambda x: x[1])[0])

        return parents

    def evolutionary_algorithm():
        """Run Evolutionary Algorithm"""
        start_time = time.time()

        pop_size = 40

        if c <= 10:  # For small problems
            pop_size = min(pop_size, math.comb(c, m))

        population = generate_population(pop_size)
        population = [(element, calculate_cost(element)) for element in population]

        elite = len(population) // 10
        if elite < 3 : elite = 3
        
        for generation in range(500):       
            next_gen = []

            population.sort(key=lambda x: -x[1])
            
            for i in range(elite): 
                next_gen.append(population[i][0].copy())
            
            while len(next_gen) < pop_size:
                p1, p2 = tournament_selection(population, 10)
                child = create_children(p1, p2)
                    
                if random.randint(1, 10) == 1:
                    child = mutate(child)
                    
                next_gen.append(child)
            
            population = [(element, calculate_cost(element)) for element in next_gen]
        
        elapsed_time = time.time() - start_time
        population.sort(key=lambda x: -x[1])
        
        best_fitness = population[0][1]
        best_solution = population[0][0]
        print(f'Final best solution: {np.sort(best_solution)}')
        print(f'Final best cost: {best_fitness:.2f}')
        
        return (best_fitness, best_solution, 1000, elapsed_time)

    # ============================================================================
    # HYBRID SEQUENTIAL
    # ============================================================================

    def hybrid_sequential():
        """Run EA then SA and return better result"""
        print("Running EA...")
        ea_fitness, ea_solution, ea_iterations, ea_time = evolutionary_algorithm()
        
        print("Running SA...")
        sa_fitness, sa_solution, sa_iterations, sa_time = simulated_annealing()
        print(f'Final best solution: {np.sort(sa_solution)}')
        print(f'Final best cost: {sa_fitness:.2f}')
        
        # Return the better result
        if sa_fitness > ea_fitness:
            best_fitness = sa_fitness
            best_solution = sa_solution
        else:
            best_fitness = ea_fitness
            best_solution = ea_solution
        
        total_iterations = ea_iterations + sa_iterations
        total_time = ea_time + sa_time
        
        print(f'\nHybrid Final best solution: {np.sort(best_solution)}')
        print(f'Hybrid Final best cost: {best_fitness:.2f}')
        
        return (best_fitness, best_solution, total_iterations, total_time)

    return hybrid_sequential()