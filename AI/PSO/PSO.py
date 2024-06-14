import numpy as np
import matplotlib.pyplot as plt

# Define the objective functions
def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

# Particle Swarm Optimization
def particle_swarm_optimization(objective_func, bounds, pop_size, max_gens, c1, c2):
    num_params = len(bounds)
    swarm_pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, num_params))
    swarm_vel = np.zeros_like(swarm_pos)
    swarm_best_pos = swarm_pos.copy()
    swarm_best_fitness = np.array([objective_func(*pos) for pos in swarm_pos])
    global_best_idx = np.argmin(swarm_best_fitness)
    global_best_pos = swarm_best_pos[global_best_idx]
    global_best_fitness = swarm_best_fitness[global_best_idx]
    best_fitnesses = []
    avg_fitnesses = []

    for gen in range(max_gens):
        for i in range(pop_size):
            # Update velocity
            swarm_vel[i] += c1 * np.random.rand() * (swarm_best_pos[i] - swarm_pos[i]) + \
                            c2 * np.random.rand() * (global_best_pos - swarm_pos[i])
            swarm_vel[i] = np.clip(swarm_vel[i], bounds[:, 0] - swarm_pos[i], bounds[:, 1] - swarm_pos[i])

            # Update position
            swarm_pos[i] += swarm_vel[i]

            # Clamp position to bounds
            swarm_pos[i] = np.clip(swarm_pos[i], bounds[:, 0], bounds[:, 1])

            # Update personal best
            fitness = objective_func(*swarm_pos[i])
            if fitness < swarm_best_fitness[i]:
                swarm_best_pos[i] = swarm_pos[i]
                swarm_best_fitness[i] = fitness

        # Update global best
        global_best_idx = np.argmin(swarm_best_fitness)
        if swarm_best_fitness[global_best_idx] < global_best_fitness:
            global_best_pos = swarm_best_pos[global_best_idx]
            global_best_fitness = swarm_best_fitness[global_best_idx]

        best_fitnesses.append(global_best_fitness)
        avg_fitness = np.mean(swarm_best_fitness)
        avg_fitnesses.append(avg_fitness)

    return global_best_pos, global_best_fitness, best_fitnesses, avg_fitnesses

# Define the bounds for each function
eggholder_bounds = np.array([[-512, 512], [-512, 512]])
holder_table_bounds = np.array([[-10, 10], [-10, 10]])

# Define parameters
pop_sizes = [20, 50, 100, 200]
num_gens = [50, 100, 200]
c1 = 2  # Cognitive parameter
c2 = 2  # Social parameter

# Plot convergence characteristics for PSO
def plot_convergence(title, avg_fitnesses, best_fitnesses, best_solution, filename, function_name, pop_size, max_gens):
    plt.plot(range(len(avg_fitnesses)), avg_fitnesses, label='Average Fitness')
    plt.plot(range(len(best_fitnesses)), best_fitnesses, label='Best Fitness')
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"{function_name} - Population Size: {pop_size}, Generations: {max_gens}")
    print(f"{function_name} - Best Fitness: {best_fitnesses[-1]}")
    print(f"{function_name} - Best Solution: {best_solution}")

# Optimize Eggholder function using PSO and plot convergence
print("Eggholder Function Optimization Results (Particle Swarm Optimization):")
for pop_size in pop_sizes:
    for max_gens in num_gens:
        best_candidate, best_fitness, avg_fitnesses, best_fitnesses = particle_swarm_optimization(eggholder, eggholder_bounds, pop_size, max_gens, c1, c2)
        title = f"Eggholder Function Convergence\nPopulation Size: {pop_size}, Generations: {max_gens}"
        filename = f"eggholder_convergence_pop{pop_size}_gens{max_gens}.png"
        plot_convergence(title, avg_fitnesses, best_fitnesses, best_candidate, filename, "Eggholder", pop_size, max_gens)

# Optimize Holder Table function using PSO and plot convergence
print("\nHolder Table Function Optimization Results (Particle Swarm Optimization):")
for pop_size in pop_sizes:
    for max_gens in num_gens:
        best_candidate, best_fitness, avg_fitnesses, best_fitnesses = particle_swarm_optimization(holder_table, holder_table_bounds, pop_size, max_gens, c1, c2)
        title = f"Holder Table Function Convergence\nPopulation Size: {pop_size}, Generations: {max_gens}"
        filename = f"holder_table_convergence_pop{pop_size}_gens{max_gens}.png"
        plot_convergence(title, avg_fitnesses, best_fitnesses, best_candidate, filename, "Holder Table", pop_size, max_gens)
