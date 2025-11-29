import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from clustering_channel_assignment import assign_clusters_distance_based, dynamic_select
from simulation import run_simulation, generate_devices, generate_arrivals, check_collision, PACKET_DURATION, TOTAL_PACKETS, NUM_CHANNELS
from simulation import reset_channel_stats
from simulation import plot_success_by_cluster_size, plot_success_by_device_count, plot_success_cdf
# ---------------------------------------------------- 
if __name__ == "__main__":

    device_counts = [200, 500, 1000, 2000, 5000,10000, 20000]
    cluster_sizes = [1, 2, 4]

    # Dynamic channel selection
    def select_channel(allowed, now):
        return dynamic_select(allowed, now)

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
            clusters, cluster_channels = assign_clusters_distance_based(d, RX, C)

            reset_channel_stats()    # VERY IMPORTANT

            prob = run_simulation(N, clusters, cluster_channels, select_channel)

            results[N][C] = prob

            print(f"N = {N}, C = {C} => Success = {prob:.4f}")

    # ---- PLOT 1: SUCCESS VS CLUSTER SIZE (for each N) ----
    plot_success_by_cluster_size(results)

    # ---- PLOT 2: SUCCESS VS DEVICE COUNT (for each C) ----
    plot_success_by_device_count(results)

    # ---- OPTIONAL: CDF PLOT FOR LAST SIMULATION ----
    plot_success_cdf()            
