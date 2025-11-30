import numpy as np
import random

#  Hill climbing local search algorithm

# Read file 
file = open("tourism_500.txt", 'r')

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


# Generate a random beginning solution
# A soultion is represented by binary list, where 1 means the landmark is included and 0 means it isnt 
def find_random_start() : 
    solution = np.zeros(m)

    for i in range(len(solution)): 
        index = random.randint(1, c)
        while index in solution: 
            index = random.randint(1, c)
        
        solution[i] = index

    return solution



"""
# Calculate cost of a solution 
def calculate_cost(sol):
    indicies = np.where(sol == 1)[0]

    names = []
    for i in range(len(indicies)): 
        names.append('e' + str(indicies[i] + 1))

    sum = 0; 
    for i in range(len(lines)): 
        if lines[i][0] in names and lines[i][1] in names: 
            sum += float(lines[i][2])

    return sum / m

"""

# Calculate cost of a solution 
def calculate_cost(sol):
    sum = 0
    for i in range(m):
        for j in range(i + 1, m):
            idx_i = int(sol[i] - 1)
            idx_j = int(sol[j] - 1)
            sum += distances[idx_i][idx_j]

    return sum / m

# Find neighbors
# To find neighbors we have to change 1 of the landmarks
def find_neighbors(sol): 
    sol_set = set(sol)
    
    for i in range(len(sol)):
        for j in range(1, c + 1):
            if j not in sol_set:
                neighbor = sol.copy()
                neighbor[i] = j
                yield neighbor



# Run the hill climbing search algorithm 
def hill_climbing(): 
    solution = find_random_start()

    it = 0
    while True : 
        it += 1
        # calculate cost of current solution
        cost = calculate_cost(solution)

        # find neighbors
        neighbors = find_neighbors(solution)

        # find the cost of each of the neighbors
        cost_neighbors = np.zeros(len(neighbors))

        for i in range(len(neighbors)):
            cost_neighbors[i] = (calculate_cost(neighbors[i]))


        index = np.argmax(cost_neighbors)
        if cost_neighbors[index] > cost: 
            solution = neighbors[index]
        else: 
            break
    return solution
"""
def find_better_neighbor(sol): 
    current_cost = calculate_cost(sol)
    neighbor_cost = 0
    neighbor = sol.copy()

    for i in range(len(sol)):
        for j in range(1, c + 1):
            if j not in sol:
                neighbor = sol.copy()
                neighbor[i] = j

                neighbor_cost = calculate_cost(neighbor)

                if neighbor_cost > current_cost: 
                    return neighbor
                    
    return sol

"""
def find_better_neighbor(sol): 
    current_cost = calculate_cost(sol)
    current_best = sol

    with multiprocessing.Pool(processes=4) as pool:
        for i in range(len(sol)//4 + 1):
            for j in range(1, c + 1):
                if j not in sol:
                    neighbor1 = sol.copy()
                    neighbor2 = sol.copy()
                    neighbor3 = sol.copy()
                    neighbor4 = sol.copy()
                    
                    neighbor1[i] = j
                    neighbor2[len(sol)//4 + i] = j
                    neighbor3[2*len(sol)//4 + i] = j
                    neighbor4[3*len(sol)//4 + i] = j

                    cost1 = pool.apply_async(calculate_cost, args=(neighbor1,))     # calculate_cost(neighbor1)
                    cost2 = pool.apply_async(calculate_cost, args=(neighbor2,))     # calculate_cost(neighbor2)
                    cost3 = pool.apply_async(calculate_cost, args=(neighbor3,))     # calculate_cost(neighbor2)
                    cost4 = pool.apply_async(calculate_cost, args=(neighbor4,))     # calculate_cost(neighbor2)

                    best = np.argmax([cost1.get(), cost2.get(), cost3.get(), cost4.get()])

                    if cost1.get() > current_cost and best == 0: 
                        current_best, current_cost = neighbor1, cost1.get()

                    elif cost2.get() > current_cost and best == 1: 
                        current_best, current_cost = neighbor2, cost2.get()
                    
                    elif cost3.get() > current_cost and best == 2: 
                        current_best, current_cost = neighbor3, cost3.get()
                    
                    elif cost4.get() > current_cost and best == 3: 
                        current_best, current_cost = neighbor4, cost4.get()
                    
                    
    return current_best, current_cost

# Run the hill climbing search algorithm 
def hill_climbing():
    current = find_random_start()
    cost_current = calculate_cost(current)

    it = 0
    while True: 
        neighbor, cost_neighbor = find_better_neighbor(current)

        it += 1
        print(f'{it}. Iteration: current: {cost_current:.2f}, neighbor: {cost_neighbor:.2f}')

        if cost_neighbor > cost_current: 
            current = neighbor
            cost_current = cost_neighbor

        elif cost_neighbor <= cost_current: 
            return current


        """
        
        # Find all neighbors and their costs
        neighbors = find_neighbors(current)
        costs = np.array([calculate_cost(n) for n in neighbors])

        # Find best neighbor
        best_idx = np.argmax(costs)
        best_cost = costs[best_idx]
        
        # If best neighbor is better, move to it
        if best_cost > cost_current:
            current = neighbors[best_idx]
        else:
            # No improving neighbor found - local optimum reached
            break
            
            """

if __name__ == '__main__':
    solution = hill_climbing()
    print(f'Final solution: {solution}')
    print(f'Final cost: {calculate_cost(solution):.2f}')


# Close the file
file.close()