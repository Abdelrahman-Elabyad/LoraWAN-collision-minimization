import os
from genetic_cluster_optimizer import optimize_clusters_with_GA, USE_MULTIPROCESSING
from simulation import run_simulation, generate_devices, generate_arrivals, reset_channel_stats
from clustering_channel_assignment import assign_clusters_random, assign_clusters_quantile_stratified, random_select

# ------------------------------------------------------------
# GA RUN PLAN
# ------------------------------------------------------------
RUN_PLAN = {
    6000:  90,     # example: run GA once for N=6000
    #20000: 3,     # example: run GA once for N=20000
}

CLUSTER_SIZES = [1]

if __name__ == "__main__":
    print("\n========== GA OPTIMIZATION STARTED ==========")
    print(f"Multiprocessing Enabled: {USE_MULTIPROCESSING}")
    print("---------------------------------------------------\n")

    run_id = 0

    for N, repetitions in RUN_PLAN.items():

        for C in CLUSTER_SIZES:

            for rep in range(repetitions):
                run_id += 1
                print(f"\n>>> GA RUN {run_id}: N={N}, C={C}, attempt {rep+1}/{repetitions}")
                print("---------------------------------------------------")

                # --------------------------------------------
                # 1. Generate devices + arrivals once
                # --------------------------------------------
                print("[INFO] Generating devices and arrivals...")
                d, RX = generate_devices(N)
                arrivals = generate_arrivals(N)
                dev_sequence = __import__("numpy").random.randint(0, N, size=len(arrivals))

                # --------------------------------------------
                # 2. Compute Baseline: RandFull (random clusters)
                # --------------------------------------------
                print("[INFO] Evaluating RandFull baseline...")
                clusters_rand, cluster_channels_rand = assign_clusters_random(N, C)
                reset_channel_stats()
                randfull_score = run_simulation(
                    N, d, RX, clusters_rand, cluster_channels_rand,
                    random_select, arrivals, dev_sequence
                )
                print(f"[Baseline RandFull] Success = {randfull_score:.4f}")

                # --------------------------------------------
                # 3. Compute Baseline: Quantile-stratified
                # --------------------------------------------
                print("[INFO] Evaluating Quantile-stratified baseline...")
                clusters_quant, cluster_channels_quant = assign_clusters_quantile_stratified(d, RX, C)
                reset_channel_stats()
                quantile_score = run_simulation(
                    N, d, RX, clusters_quant, cluster_channels_quant,
                    random_select, arrivals, dev_sequence
                )
                print(f"[Baseline Quantile] Success = {quantile_score:.4f}")

                # --------------------------------------------
                # 4. Run GA Optimization
                # --------------------------------------------
                print("[INFO] Running Genetic Algorithm...")
                best_genome, best_fitness = optimize_clusters_with_GA(N=N, C=C)

                # --------------------------------------------
                # 5. FINAL COMPARISON
                # --------------------------------------------
                print("\n================ FINAL COMPARISON ================")
                print(f"N = {N}, C = {C}")
                print("-----------------------------------------------")
                print(f" RandFull Baseline:       {randfull_score:.4f}")
                print(f" Quantile Baseline:       {quantile_score:.4f}")
                print(f" GA Best Genome Fitness:  {best_fitness:.4f}")
                print("-----------------------------------------------")
                print(f" Improvement over RandFull:  {best_fitness - randfull_score:.4f}")
                print(f" Improvement over Quantile:  {best_fitness - quantile_score:.4f}")
                print(f" Relative Improvement:       {100*(best_fitness - randfull_score)/randfull_score:.2f}%")
                print("===================================================\n")

    print("\n========== GA OPTIMIZATION COMPLETED ==========\n")
