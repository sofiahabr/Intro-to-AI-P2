import numpy as np
import random
import time
import math

def run(test_file): 
    """
    Hybrid: SA Embedded in EA
    
    This approach:
    1. Runs Evolutionary Algorithm normally
    2. Each generation, refines elite individuals with Simulated Annealing
    3. Uses improved solutions in next generation
    4. Results in better overall solutions
    
    This is more sophisticated than sequential hybrids.
    """
    start_time = time.time()
    
    # Read file 
    file = open(test_file, 'r')
    line_list = file.readlines()
    file.close()

    c, m = line_list[0].split()
    c = int(c)
    m = int(m)

    line_list.pop(0)
    lines = []

    for i in range(len(line_list)):
        lines.append(line_list[i].split())

    # Create distance matrix
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
        if len(set(sol)) != m: 
            return 0
        
        total_cost = 0
        for i in range(m):
            for j in range(i + 1, m):
                idx_i = sol[i] - 1
                idx_j = sol[j] - 1
                total_cost += distances[idx_i][idx_j]
        
        return total_cost / m

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW: SA REFINEMENT FUNCTION (embedded in EA)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def sa_refinement(solution, max_iterations=50):
        """
        Apply Simulated Annealing to refine a solution.
        Used to improve elite individuals during EA generations.
        """
        current = solution.copy()
        current_cost = calculate_cost(current)
        best = current.copy()
        best_cost = current_cost
        
        temperature = 10.0
        cooling_rate = 0.95
        
        for iteration in range(max_iterations):
            # Try a neighbor (swap one element)
            neighbor = current.copy()
            idx = random.randint(0, m - 1)
            neighbor[idx] = random.randint(1, c)
            
            neighbor_cost = calculate_cost(neighbor)
            
            # Accept if better
            if neighbor_cost > current_cost:
                current = neighbor.copy()
                current_cost = neighbor_cost
                
                # Update best if even better
                if current_cost > best_cost:
                    best = current.copy()
                    best_cost = current_cost
            else:
                # Accept worse with SA probability
                delta = neighbor_cost - current_cost
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.01)):
                    current = neighbor.copy()
                    current_cost = neighbor_cost
            
            # Cool down temperature
            temperature *= cooling_rate
        
        return best, best_cost
    

    # One point crossover 
    def create_children(parent1, parent2):
        child = np.zeros(m, dtype=int)
        for i in range(m//2): 
            child[i] = parent1[i]
        for i in range(m//2, m): 
            child[i] = parent2[i]
        return child

    # Mutation
    def mutate(sol):
        sol = sol.copy()
        index = random.randint(0, m - 1)
        random_replacement = random.randint(1, c)
        sol[index] = random_replacement
        return sol

    # Generate random solution
    def generate_random_solution():
        return np.random.choice(c, m, replace=False) + 1

    # Generate population
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

    # Roulette wheel selection
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

    # ═══════════════════════════════════════════════════════════════════════════
    # HYBRID EA WITH EMBEDDED SA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def hybrid_evolutionary_algorithm():
        pop_size = 40
        population = generate_population(pop_size)
        population = [(element, calculate_cost(element)) for element in population]

        par = len(population) // 2
        print(f'Amount of parents: {par}')

        elite = len(population) // 10
        if elite < 3: 
            elite = 3
        print(f'Amount of elite: {elite}')
        
        for generation in range(1000):
            next_gen = []

            population.sort(key=lambda x: -x[1])
            
            # ════════════════════════════════════════════════════════════════
            # HYBRID STEP: Refine elite with Simulated Annealing
            # ════════════════════════════════════════════════════════════════
            for i in range(elite):
                # Apply SA to improve this elite individual
                improved_sol, improved_cost = sa_refinement(
                    population[i][0], 
                    max_iterations=40  # Tune this: 30-100 iterations
                )
                # Replace original with improved version
                next_gen.append(improved_sol)
            
            # Generate children through crossover and mutation (normal EA)
            while len(next_gen) < pop_size:
                p1 = roulette_wheel_selection(population)
                p2 = roulette_wheel_selection(population)
                
                child = create_children(p1, p2)
                    
                # 10% mutation rate
                if random.randint(1, 10) == 1:
                    child = mutate(child)
                    
                next_gen.append(child)
            
            # Evaluate new population
            population = [(element, calculate_cost(element)) for element in next_gen]
        
        population.sort(key=lambda x: -x[1])
        return population[0][0], population[0][1]

    best_solution, best_fitness = hybrid_evolutionary_algorithm()
    
    elapsed_time = time.time() - start_time

    print(f'\nFinal best solution: {best_solution}')
    print(f'Final best cost: {best_fitness:.2f}')

    # Return in correct format: (fitness, solution, iterations, elapsed_time)
    return (best_fitness, best_solution, 1000, elapsed_time)


if __name__ == "__main__":
    # For testing
    result = run("data/tourism_5.txt")
    print(f"\nResult: fitness={result[0]:.2f}, solution_size={len(result[1])}, time={result[3]:.2f}s")