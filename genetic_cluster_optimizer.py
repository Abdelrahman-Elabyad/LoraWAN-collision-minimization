import numpy as np
import csv
import os
import multiprocessing as mp
from simulation import generate_arrivals, run_simulation, generate_devices, reset_channel_stats
from clustering_channel_assignment import random_select
from clustering_channel_assignment import assign_clusters_quantile_stratified
# ----------------------------------------------------
# Genetic Algorithm Parameters
# ----------------------------------------------------
POP_SIZE = 20              # Population size
GENERATIONS = 30           # Number of generations
TOURNAMENT_SIZE = 4        # Tournament selection group size
MUTATION_RATE = 0.01       # Probability of mutating each gene
ELITE_COUNT = 2            # Number of survivors per generation
TOTAL_CHANNELS = 8


USE_MULTIPROCESSING = True    # ← start with False

# ----------------------------------------------------
# Create Cluster → Channels Mapping
# ----------------------------------------------------
def build_cluster_channels(C):
    K = TOTAL_CHANNELS // C  # number of clusters
    cluster_channels = []
    for k in range(K):
        start = k * C
        cluster_channels.append(list(range(start, start + C)))
    return cluster_channels, K


# ----------------------------------------------------
# Fitness Function
# ----------------------------------------------------
def evaluate_genome(genome,cluster_channels, N, d, RX, C,arrivals,dev_sequence):
    success_prob = run_simulation(N,d,RX,genome,cluster_channels,random_select,arrivals,dev_sequence)
    return success_prob

def _eval_worker(args):
    """
    Worker for multiprocessing: args is a tuple.
    """
    genome, cluster_channels, N, d, RX, C, arrivals, dev_sequence = args
    return evaluate_genome(genome, cluster_channels, N, d, RX, C, arrivals, dev_sequence)
# ----------------------------------------------------
# Create Initial Population
# ----------------------------------------------------

def initialize_population(N, K, d, RX, C):
    """
    Hybrid initialization:
    - 20% quantile-stratified seeds
    - 80% random genomes
    """

    population = []

    num_stratified = max(2, POP_SIZE // 5)   # 20% stratified
    num_random = POP_SIZE - num_stratified

    # ---- 1. Generate quantile-stratified seeds ----
    clusters, _ = assign_clusters_quantile_stratified(d, RX, C)

    population.append(clusters.copy())

    # Create slight mutations of the stratified genome
    for _ in range(num_stratified - 1):
        mutated = clusters.copy()
        # Mutate ~1% of genes
        num_mut = max(1, int(0.01 * N))
        idxs = np.random.choice(N, num_mut, replace=False)
        for i in idxs:
            mutated[i] = np.random.randint(0, K)
        population.append(mutated)

    # ---- 2. Fill rest with pure random genomes ----
    for _ in range(num_random):
        pop = np.random.randint(0, K, size=N, dtype=int)
        population.append(pop)

    return population


# ----------------------------------------------------
# Tournament Selection
# ----------------------------------------------------
def tournament_select(pop, fitness):
    idxs = np.random.choice(len(pop), TOURNAMENT_SIZE)
    best_idx = idxs[np.argmax([fitness[i] for i in idxs])]
    return pop[best_idx]


# ----------------------------------------------------
# One-Point Crossover
# ----------------------------------------------------
def crossover(parent1, parent2):
    N = len(parent1)
    point = np.random.randint(1, N - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


# ----------------------------------------------------
# Mutation
# ----------------------------------------------------
def mutate(child, K, max_mut=5):
    N = len(child)
    idxs = np.random.choice(N, size=max_mut, replace=False)
    for i in idxs:
        child[i] = np.random.randint(0, K)
    return child

# ----------------------------------------------------
# Main GA Function
# ----------------------------------------------------
def optimize_clusters_with_GA(N=500, C=2):
    print(f"\n=== Running GA for N={N}, C={C} ===")

    # Generate device geometry
    d, RX = generate_devices(N)
    arrivals = generate_arrivals(N)
    cluster_channels, K = build_cluster_channels(C)
    dev_sequence = np.random.randint(0, N, size=len(arrivals))
    # Initialize population
    population = initialize_population(N, K, d, RX, C)

    best_fitness = -1
    best_genome = None

    # GA Evolution Loop
    for gen in range(GENERATIONS):

        if USE_MULTIPROCESSING:
            # Build argument list for each genome
            args_list = [
                (genome, cluster_channels, N, d, RX, C, arrivals, dev_sequence)
                for genome in population
            ]

            with mp.Pool() as pool:
                fitness = pool.map(_eval_worker, args_list)
        else:
            # Single-core fallback (current behavior)
            fitness = []
            for idx, genome in enumerate(population):
                f = evaluate_genome(genome, cluster_channels, N, d, RX, C, arrivals, dev_sequence)
                fitness.append(f)

        # Track best genome
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_genome = population[gen_best_idx].copy()

        print(f"Generation {gen+1}/{GENERATIONS} → Best Fitness = {best_fitness:.4f}")

        # ELITISM: keep top elite genomes
        sorted_idx = np.argsort(fitness)[::-1]
        new_population = [population[i].copy() for i in sorted_idx[:ELITE_COUNT]]

        # Create children until population full
        while len(new_population) < POP_SIZE:
            parent1 = tournament_select(population, fitness)
            parent2 = tournament_select(population, fitness)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, K)
            child2 = mutate(child2, K)

            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]

        

        print("\n======= GA Finished =======")
    print(f"Generation {gen+1}/{GENERATIONS} → "
          f"GenBest = {max(fitness):.4f}, "
          f"GlobalBest = {best_fitness:.4f}")

    # ==========================================================
    # SAVE BEST GENOME (optional)
    # ==========================================================
    np.save(f"best_genome_N{N}_C{C}.npy", best_genome)
    print(f"[SAVED] best_genome_N{N}_C{C}.npy")

    # ==========================================================
    # APPEND TRAINING DATA TO MASTER DATASET
    # ==========================================================
    master_csv = "GA_dataset_master.csv"

    write_header = not os.path.exists(master_csv)

    with open(master_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["N", "device_id", "d", "RX", "C", "cluster_optimal"])

        for i in range(N):
            writer.writerow([N, i, float(d[i]), float(RX[i]), C, int(best_genome[i])])

    print(f"[APPENDED] Best genome data added to {master_csv}")

    print("\n== GA Completed Successfully ==")

    return best_genome, best_fitness
