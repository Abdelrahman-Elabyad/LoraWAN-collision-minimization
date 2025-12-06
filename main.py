import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math
from clustering_channel_assignment import assign_clusters_quantile_stratified, dynamic_select, random_select
from simulation import run_simulation, generate_devices, generate_arrivals, check_collision, PACKET_DURATION, TOTAL_PACKETS, NUM_CHANNELS, channel_stats, DATASET_FILE, USE_ML
from simulation import reset_channel_stats
from simulation import plot_success_by_cluster_size, plot_success_by_device_count, plot_success_cdf
import subprocess

# ----------------------------------------------------
# ML Enalbled or Disabled
# ----------------------------------------------------

if __name__ == "__main__":

    device_counts = [200, 500, 1000, 2000, 5000,10000, 20000, 50000, 100000]
    cluster_sizes = [1, 2, 4, 8]

    # Dynamic channel selection
    def select_channel(allowed, now, observed_stats):
        return dynamic_select(allowed, now, observed_stats)

    # Dictionary to store results: results[N][C] = success probability
    results = {}

    # ---- RUN ALL EXPERIMENTS ----
    for N in device_counts:
        results[N] = {}

        print(f"\n========== Testing N = {N} devices ==========")

        # Generate devices ONCE for each N
        d, RX = generate_devices(N)

        for C in cluster_sizes:
            print(f"--- Running C = {C} ---")

            # Static clustering
            clusters, cluster_channels = assign_clusters_quantile_stratified(d, RX, C)

            # ==============================
            # 1) Dynamic (TS) Selection
            # ==============================
            reset_channel_stats()
            prob_ts = run_simulation(N, d, RX, clusters, cluster_channels, select_channel)

            results[N][C] = prob_ts   # keep TS version for plots

            # ==============================
            # 2) Random Channel Selection
            # ==============================
            reset_channel_stats()
            prob_rand = run_simulation(N,d, RX, clusters, cluster_channels, random_select)
            print(
                f"[RANDOM] N={N}, C={C} => "
                f"Success = {prob_rand:.4f},  "
            )
        
    # ---- PLOT 1: SUCCESS VS CLUSTER SIZE (for each N) ----
    plot_success_by_cluster_size(results)

    # ---- PLOT 2: SUCCESS VS DEVICE COUNT (for each C) ----
    plot_success_by_device_count(results)

    # ---- OPTIONAL: CDF PLOT FOR LAST SIMULATION ----
    plot_success_cdf()    
    # ---- TRAIN BEST PARAMETERS ----
    if USE_ML:
        subprocess.run(["python", "train_parameters.py"])
        print("ML training complete. Updated best_params.json loaded next run.")

        
