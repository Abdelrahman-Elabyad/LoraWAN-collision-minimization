# ============================================
# main_ga.py — Automated GA Dataset Collector
# ============================================

import os
from genetic_cluster_optimizer import optimize_clusters_with_GA, USE_MULTIPROCESSING

# ------------------------------------------------------------
# RUN PLAN — how many GA runs you want for each device count N
# ------------------------------------------------------------
RUN_PLAN = {
    #200:   5,
    #500:   5,
    #1000:  5,
    #2000:  4,
    #5000:  3,
    10000: 5,
    #20000: 1,
}

# ------------------------------------------------------------
# Cluster sizes to consider (C ∈ {1, 2, 4})
# ------------------------------------------------------------
CLUSTER_SIZES = [1, 2, 4]

# ------------------------------------------------------------
# OPTIONAL: Clear old master dataset before starting
# ------------------------------------------------------------
MASTER_DATASET = "GA_dataset_master.csv"

# ------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n========== GA DATASET COLLECTION STARTED ==========")
    print(f"Multiprocessing Enabled: {USE_MULTIPROCESSING}")
    print("---------------------------------------------------\n")

    run_id = 0

    # Loop through defined Ns
    for N, repetitions in RUN_PLAN.items():

        # Loop through cluster sizes
        for C in CLUSTER_SIZES:

            for rep in range(repetitions):
                run_id += 1
                print(f"\n>>> RUN {run_id}: N={N}, C={C}, attempt {rep+1}/{repetitions}")

                # Run GA and populate dataset
                best_genome, best_fitness = optimize_clusters_with_GA(N=N, C=C)

                print(f"*** DONE RUN {run_id}: Best Fitness = {best_fitness:.4f}")
                print("---------------------------------------------------")

    print("\n========== GA DATASET COLLECTION COMPLETED ==========")
    print("Your master dataset is ready: GA_dataset_master.csv\n")
