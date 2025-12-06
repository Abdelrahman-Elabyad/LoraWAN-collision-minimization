import numpy as np
import math
from clustering_channel_assignment import assign_clusters_quantile_stratified, dynamic_select, random_select
from simulation import run_simulation, generate_devices, reset_channel_stats
from simulation import plot_success_by_cluster_size, plot_success_by_device_count, plot_success_cdf
from simulation import PACKET_DURATION, MEAN_INTERARRIVAL, NUM_CHANNELS

def pure_aloha_theory(N):
    """
    Calculate Pure ALOHA success probability.
    G = (arrival_rate_per_channel) × (packet_duration)
    G = (N/MEAN_INTERARRIVAL/NUM_CHANNELS) × PACKET_DURATION
    """
    arrival_rate = N / MEAN_INTERARRIVAL  # total packets/second
    rate_per_channel = arrival_rate / NUM_CHANNELS
    G = rate_per_channel * PACKET_DURATION
    return math.exp(-2 * G)

if __name__ == "__main__":

    device_counts = [200, 500, 1000, 2000, 5000, 10000, 20000]
    cluster_sizes = [1, 2, 4, 8]
    results = {}

    for N in device_counts:
        results[N] = {}
        print(f"\n========== N = {N} ==========")
        
        d, RX = generate_devices(N)
        theory = pure_aloha_theory(N)

        for C in cluster_sizes:
            clusters, cluster_channels = assign_clusters_quantile_stratified(d, RX, C)

            reset_channel_stats()
            prob_random = run_simulation(N, d, RX, clusters, cluster_channels, random_select)

            reset_channel_stats()
            prob_dynamic = run_simulation(N, d, RX, clusters, cluster_channels, dynamic_select)

            results[N][C] = prob_dynamic

            print(f"C={C}: Random={prob_random:.4f}, Dynamic={prob_dynamic:.4f}", end="")
            
            if C == 8:
                print(f", Theory={theory:.4f}, Ratio={prob_random/theory:.2f}x", end="")
            print()

    plot_success_by_cluster_size(results)
    plot_success_by_device_count(results)
    plot_success_cdf()