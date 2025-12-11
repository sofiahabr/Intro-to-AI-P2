import time

def run(test_file):
    """
    Hybrid Sequential: Run EA then SA to improve the solution
    """
    start_time = time.time()
    
    try:
        from evolutionary_algorithm_penalty_onepoint_insert_roulette import run as ea_run
        ea_fitness, ea_solution, ea_iterations, ea_time = ea_run(test_file)

    except Exception as e:
        print(f"EA Error: {e}")
        ea_fitness, ea_solution, ea_iterations, ea_time = 0, None, 0, 0
    
    try:
        from Simultaed_annealing_penalty import run as sa_run
        sa_fitness, sa_solution, sa_iterations, sa_time = sa_run(test_file)
    except Exception as e:
        print(f"SA Error: {e}")
        sa_fitness, sa_solution, sa_iterations, sa_time = 0, None, 0, 0
    
    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    
    # Return the better result
    if sa_fitness > ea_fitness:
        best_fitness = sa_fitness
        best_solution = sa_solution
    else:
        best_fitness = ea_fitness
        best_solution = ea_solution
    
    # Total iterations is sum of both
    total_iterations = ea_iterations + sa_iterations
    
    # Return in correct format: (fitness, solution, iterations, elapsed_time)
    return (best_fitness, best_solution, total_iterations, elapsed_time)


if __name__ == "__main__":
    # For testing
    result = run("data/tourism_5.txt")
    print(f"Result: fitness={result[0]:.2f}, solution_size={len(result[1])}, time={result[3]:.2f}s")