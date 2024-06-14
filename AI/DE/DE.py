import numpy as np
import matplotlib.pyplot as plt

# Define the objective functions
def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

# Differential Evolution Optimization
def differential_evolution(objective_func, bounds, pop_size, max_gens, crossover_prob=0.8):
    num_params = len(bounds)
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, num_params))
    best_candidate = None
    best_fitness = float('inf')
    best_fitnesses = []
    avg_fitnesses = []

    for gen in range(max_gens):
        new_pop = np.zeros_like(pop)

        for i in range(pop_size):
            target_vector = pop[i]
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + 0.5 * (b - c), bounds[:, 0], bounds[:, 1])

            # Crossover
            j_rand = np.random.randint(num_params)
            trial_vector = np.array([mutant_vector[j] if np.random.rand() < crossover_prob or j == j_rand else target_vector[j] for j in range(num_params)])

            # Selection
            if objective_func(*trial_vector) < objective_func(*target_vector):
                pop[i] = trial_vector

                if objective_func(*trial_vector) < best_fitness:
                    best_candidate = trial_vector
                    best_fitness = objective_func(*trial_vector)
            else:
                if objective_func(*target_vector) < best_fitness:
                    best_candidate = target_vector
                    best_fitness = objective_func(*target_vector)

        best_fitnesses.append(best_fitness)
        avg_fitness = np.mean([objective_func(*candidate) for candidate in pop])
        avg_fitnesses.append(avg_fitness)

    return best_candidate, best_fitness, best_fitnesses, avg_fitnesses

# Define the bounds for each function
eggholder_bounds = np.array([[-512, 512], [-512, 512]])
holder_table_bounds = np.array([[-10, 10], [-10, 10]])

# Define parameters
pop_sizes = [20, 50, 100, 200]
num_gens = [50, 100, 200]

# Plot convergence characteristics for DE
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

# Optimize Eggholder function using DE and plot convergence
print("Eggholder Function Optimization Results (Differential Evolution):")
for pop_size in pop_sizes:
    for max_gens in num_gens:
        best_candidate, best_fitness, avg_fitnesses, best_fitnesses = differential_evolution(eggholder, eggholder_bounds, pop_size, max_gens)
        title = f"Eggholder Function Convergence\nPopulation Size: {pop_size}, Generations: {max_gens}"
        filename = f"eggholder_convergence_pop{pop_size}_gens{max_gens}.png"
        plot_convergence(title, avg_fitnesses, best_fitnesses, best_candidate, filename, "Eggholder", pop_size, max_gens)

# Optimize Holder Table function using DE and plot convergence
print("\nHolder Table Function Optimization Results (Differential Evolution):")
for pop_size in pop_sizes:
    for max_gens in num_gens:
        best_candidate, best_fitness, avg_fitnesses, best_fitnesses = differential_evolution(holder_table, holder_table_bounds, pop_size, max_gens)
        title = f"Holder Table Function Convergence\nPopulation Size: {pop_size}, Generations: {max_gens}"
        filename = f"holder_table_convergence_pop{pop_size}_gens{max_gens}.png"
        plot_convergence(title, avg_fitnesses, best_fitnesses, best_candidate, filename, "Holder Table", pop_size, max_gens)
