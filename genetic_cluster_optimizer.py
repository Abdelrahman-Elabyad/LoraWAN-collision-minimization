import numpy as np
import csv
import os
import multiprocessing as mp
from simulation import generate_arrivals, run_simulation, generate_devices, reset_channel_stats
from clustering_channel_assignment import random_select
from clustering_channel_assignment import assign_clusters_quantile_stratified
import pandas as pd
# ----------------------------------------------------
# Genetic Algorithm Parameters
# ----------------------------------------------------

TOTAL_CHANNELS = 8


USE_MULTIPROCESSING = True    # ← start with False

def auto_select_mode(N):
    if N <= 8000:
        return "balanced"
    else:
        return "exploratory"

def get_ga_params(N, GA_MODE):
    
    if N <= 300:
        base = {"POP_SIZE": 50, "GENERATIONS": 100, "MAX_MUT": 3, "MUT_RATE": 0.01, "TOURNAMENT": 3, "ELITE": 2}

    elif N <= 800:
        base = {"POP_SIZE": 50, "GENERATIONS": 100, "MAX_MUT": 4, "MUT_RATE": 0.01, "TOURNAMENT": 3, "ELITE": 2}

    elif N <= 2000:
        base = {"POP_SIZE": 50, "GENERATIONS": 150, "MAX_MUT": 6, "MUT_RATE": 0.008, "TOURNAMENT": 4, "ELITE": 2}

    elif N <= 6000:
        base = {"POP_SIZE": 100, "GENERATIONS": 300, "MAX_MUT": 10, "MUT_RATE": 0.006, "TOURNAMENT": 4, "ELITE": 2}

    elif N <= 12000:
        base = {"POP_SIZE": 100, "GENERATIONS": 400, "MAX_MUT": 12, "MUT_RATE": 0.005, "TOURNAMENT": 5, "ELITE": 2}

    else:  # N > 12000
        base = {"POP_SIZE": 100, "GENERATIONS": 500, "MAX_MUT": 20, "MUT_RATE": 0.004, "TOURNAMENT": 5, "ELITE": 3}

    # ---------- MODE PARAMETERS ----------
    if GA_MODE == "balanced":
        base.update({"R_NORMAL": 0.80, "R_STRONG": 0.15, "R_RANDOM": 0.05})
    elif GA_MODE == "exploratory":
        base.update({"R_NORMAL": 0.60, "R_STRONG": 0.25, "R_RANDOM": 0.15})
    else:  # fallback
        base.update({"R_NORMAL": 1.0, "R_STRONG": 0.0, "R_RANDOM": 0.0})

    return base


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
    success_prob, _, _ = run_simulation(N,d,RX,genome,cluster_channels,random_select,arrivals,dev_sequence)
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

def initialize_population(N, K, d, RX, C, POP_SIZE, MUT_RATE, MAX_MUT):
    """
    Hybrid initialization:
    - 20% quantile-stratified seeds (plus light adaptive mutation)
    - 80% random genomes
    """
    population = []

    num_stratified = max(2, POP_SIZE // 5)   # 20%
    num_random = POP_SIZE - num_stratified

    # ---------- 1. Base stratified genome ----------
    clusters= assign_clusters_quantile_stratified(d, RX, C)
    population.append(clusters.copy())

    # ---------- 2. Mutated versions of the stratified seed ----------
    for _ in range(num_stratified - 1):
        mutated = clusters.copy()

        # Apply mutation with probability MUT_RATE
        if np.random.rand() < MUT_RATE:
            # Mutate MAX_MUT genes
            idxs = np.random.choice(N, size=min(MAX_MUT, N), replace=False)
            for i in idxs:
                mutated[i] = np.random.randint(0, K)

        population.append(mutated)

    # ---------- 3. Random genomes ----------
    for _ in range(num_random):
        genome = np.random.randint(0, K, size=N)
        population.append(genome)

    return population


# ----------------------------------------------------
# Tournament Selection
# ----------------------------------------------------
def tournament_select(pop, fitness,TOURNAMENT_SIZE):
    idxs = np.random.choice(len(pop), TOURNAMENT_SIZE)
    best_idx = idxs[np.argmax([fitness[i] for i in idxs])]
    return pop[best_idx]


# ----------------------------------------------------
# One-Point Crossover
# ----------------------------------------------------
def crossover(parent1, parent2):
    N = len(parent1)
    if N <= 2:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, N - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


# ----------------------------------------------------
# Mutation
# ----------------------------------------------------
def structural_mutation_swap(child):
    """Swap two genes (devices) between clusters."""
    N = len(child)
    i, j = np.random.randint(0, N, size=2)
    child[i], child[j] = child[j], child[i]
    return child

def structural_mutation_block(child, block_size=20):
    N = len(child)
    if N < block_size * 2:
        return child

    start1 = np.random.randint(0, N - block_size)
    start2 = np.random.randint(0, N - block_size)

    # Swap slices
    tmp = child[start1:start1+block_size].copy()
    child[start1:start1+block_size] = child[start2:start2+block_size]
    child[start2:start2+block_size] = tmp

    return child

def random_small_mutation(child, K, count=3):
    N = len(child)
    idxs = np.random.choice(N, size=min(count, N), replace=False)
    for i in idxs:
        child[i] = np.random.randint(0, K)
    return child

# ----------------------------------------------------
# Main GA Function
# ----------------------------------------------------

def optimize_clusters_with_GA(N, C, d, RX, positions, arrivals, dev_sequence):
    print(f"\n=== Running GA for N={N}, C={C} ===")

    # AUTO-SELECT MODE
    GA_MODE = auto_select_mode(N)
    print(f"Auto-selected GA MODE = {GA_MODE}")

    # GET PARAMETERS FOR THIS N + MODE
    params = get_ga_params(N, GA_MODE)

    POP_SIZE = params["POP_SIZE"]
    GENERATIONS = params["GENERATIONS"]
    MIN_GEN = GENERATIONS                           # must run at least this many
    MAX_GEN = GENERATIONS + 50                           # max generations
    EARLY_STOP_PATIENCE = 20                       # no improvement threshold
    MAX_MUT = params["MAX_MUT"]
    MUT_RATE = params["MUT_RATE"]
    TOURNAMENT_SIZE = params["TOURNAMENT"]
    ELITE_COUNT = params["ELITE"]
    PERFECT_FITNESS_THRESHOLD = 1.0
    R_NORMAL = params["R_NORMAL"]
    R_STRONG = params["R_STRONG"]
    R_RANDOM = params["R_RANDOM"]
    

    print(f"GA Parameters: POP={POP_SIZE}, GEN={GENERATIONS}, "
          f"MUT={MUT_RATE}, MAX_MUT={MAX_MUT}, "
          f"Tourn={TOURNAMENT_SIZE}, Elite={ELITE_COUNT}, "
          f"Mode={GA_MODE}")

    x = positions[:, 0]
    y = positions[:, 1]
    cluster_channels, K = build_cluster_channels(C)
    # Initialize population (now adaptive)
    population = initialize_population(N, K, d, RX, C, POP_SIZE, MUT_RATE, MAX_MUT)

    best_fitness = -1
    best_genome = None
    no_improve_count = 0
    # -------------------------------
    # GA EVOLUTION LOOP
    # -------------------------------
    for gen in range(1, MAX_GEN + 1):

        # Fitness evaluation
        if USE_MULTIPROCESSING:
            args_list = [(genome, cluster_channels, N, d, RX, C, arrivals, dev_sequence)for genome in population]
            with mp.Pool() as pool:
                fitness = pool.map(_eval_worker, args_list)
        else:
            fitness = [evaluate_genome(g, cluster_channels, N, d, RX, C, arrivals, dev_sequence)for g in population]

        # Track best genome
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_genome = population[gen_best_idx].copy()
            no_improve_count = 0          # reset counter on improvement
        else:
            no_improve_count += 1         # increment if no improvement

        print(f"Generation {gen}/{MAX_GEN} → Best Fitness = {best_fitness:.4f} "f"(no improvement = {no_improve_count})")

        # -------------------------------
        # STOPPING CONDITIONS
        # -------------------------------
        
        # Condition 1: Perfect fitness reached
        if best_fitness >= PERFECT_FITNESS_THRESHOLD:
            print(f"\n🎯 PERFECT FITNESS REACHED at Gen {gen}!")
            print(f"Fitness = {best_fitness:.4f} >= {PERFECT_FITNESS_THRESHOLD}")
            break
        
        # Condition 2: No improvement after MIN_GEN with patience exceeded
        if gen >= MIN_GEN and no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n🟡 EARLY STOPPING TRIGGERED at Gen {gen}")
            print(f"No improvement for {EARLY_STOP_PATIENCE} generations after {MIN_GEN} minimum.")
            break
        # ELITISM
        sorted_idx = np.argsort(fitness)[::-1]
        new_population = [population[i].copy() for i in sorted_idx[:ELITE_COUNT]]

        # --------------------------------------------
        # EXPLORATION-AWARE CHILD GENERATION
        # --------------------------------------------
        num_elites = ELITE_COUNT
        num_normal = int(R_NORMAL * POP_SIZE)
        num_strong = int(R_STRONG * POP_SIZE)
        num_random = POP_SIZE - (num_elites + num_normal + num_strong)

        # 1) NORMAL CHILDREN
        while len(new_population) < num_elites + num_normal:
            p1 = tournament_select(population, fitness, TOURNAMENT_SIZE)
            p2 = tournament_select(population, fitness, TOURNAMENT_SIZE)

            #create children (CROSSOVER)
            c1, c2 = crossover(p1, p2)

            # NORMAL mutation = 80% swap, 20% tiny random mutation
            if np.random.rand() < MUT_RATE:
                if np.random.rand() < 0.8:
                    c1 = structural_mutation_swap(c1)
                else:
                    c1 = random_small_mutation(c1, K, count=3)

            if np.random.rand() < MUT_RATE:
                if np.random.rand() < 0.8:
                    c2 = structural_mutation_swap(c2)
                else:
                    c2 = random_small_mutation(c2, K, count=3)


            new_population.extend([c1, c2])

        # 2) STRONG MUTATION CHILDREN
        strong_mut = min(MAX_MUT * 4, N)

        while len(new_population) < num_elites + num_normal + num_strong:
            p1 = tournament_select(population, fitness, TOURNAMENT_SIZE)
            p2 = tournament_select(population, fitness, TOURNAMENT_SIZE)

            child, _ = crossover(p1, p2)
            child = structural_mutation_block(child, block_size=MAX_MUT)
            new_population.append(child)

        # 3) RANDOM IMMIGRANTS
        while len(new_population) < POP_SIZE:
            immigrant = np.random.randint(0, K, size=N)
            new_population.append(immigrant)

        population = new_population[:POP_SIZE]

    # =============================
    # FINAL EVALUATION WITH CDF
    # =============================
    print("\n[GA] Evolution complete. Running FINAL evaluation with CDF tracking...")
    reset_channel_stats()
    final_score, success_distances, failure_distances = run_simulation(
        N, d, RX, best_genome, cluster_channels, random_select, arrivals, dev_sequence
    )

    print(f"[GA] Final evaluation score: {final_score:.4f}")
    
    if success_distances is not None:
        print(f"[GA] Success distances tracked: {len(success_distances)} packets")
        print(f"[GA] Failure distances tracked: {len(failure_distances)} packets")
    else:
        print(f"[GA] WARNING: Distance tracking returned None!")
        success_distances = []
        failure_distances = []

    # Save best genome
    np.save(f"best_genome_N{N}_C{C}.npy", best_genome)
    print(f"[SAVED] best_genome_N{N}_C{C}.npy")

    # Save to master dataset
    master_csv = "GA_dataset_master.csv"

    if os.path.exists(master_csv):
        df_tmp = pd.read_csv(master_csv)
        if "network_id" in df_tmp.columns and len(df_tmp) > 0:
            network_id = int(df_tmp["network_id"].iloc[-1]) + 1
        else:
            network_id = 0
    else:
        network_id = 0

    write_header = not os.path.exists(master_csv)

    with open(master_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "network_id", "N", "device_id",
                "d", "RX", "C", "cluster_optimal",
                "x", "y"
            ])

        for i in range(N):
            writer.writerow([
                network_id,
                N,
                i,
                float(d[i]),
                float(RX[i]),
                C,
                int(best_genome[i]),
                float(x[i]),
                float(y[i])
            ])

    print(f"[APPENDED] Added best genome data to {master_csv} with network_id={network_id}")
    print("\n== GA Completed Successfully ==\n")

    return best_genome, final_score, success_distances, failure_distances