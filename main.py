import numpy as np
import math
from clustering_channel_assignment import assign_clusters_quantile_stratified,assign_clusters_random, thompson_select_wrapper, random_select,scoring_select, smart_dynamic_select
from genetic_cluster_optimizer import build_cluster_channels
from simulation import run_simulation, generate_devices, reset_channel_stats, generate_arrivals
from simulation import plot_success_by_cluster_size, plot_success_by_device_count, plot_success_cdf
from simulation import PACKET_DURATION, MEAN_INTERARRIVAL, NUM_CHANNELS
from cluster_ml_model import load_trained_model, assign_clusters_ml

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

    device_counts = [20000]
    cluster_sizes = [1, 2, 4, 8]
    results = {}

    ml_model = load_trained_model("cluster_model_lgbm.bin")
    print("[INFO] ML model loaded.")

    for N in device_counts:
        print(f"\n========== N = {N} ==========")
        d, RX = generate_devices(N)
        theory = pure_aloha_theory(N)
        
        # SAME arrivals for all algorithms
        arrivals = generate_arrivals(N)
        dev_sequence = np.random.randint(0, N, size=len(arrivals))
        
        for C in cluster_sizes:
            cluster_channels, _ = build_cluster_channels(C)
            # ----------------------------------------------------
            # 1) TRUE RANDOM BASELINE (Random clusters + Random channels)
            # ----------------------------------------------------
            clusters_rand= assign_clusters_random(N, C)

            reset_channel_stats()
            prob_rand_full = run_simulation(N, d, RX, clusters_rand, cluster_channels,random_select, arrivals, dev_sequence)

            # ----------------------------------------------------
            # 2) SMART STATIC CLUSTERS BASELINE
            # ----------------------------------------------------
            clusters_quantile= assign_clusters_quantile_stratified(d, RX, C)

            # Random channels on smart clusters
            reset_channel_stats()
            prob_random = run_simulation(N, d, RX, clusters_quantile, cluster_channels,random_select, arrivals, dev_sequence)

            # Smart (Long-Term Learning)
            reset_channel_stats()
            prob_smart_long_term = run_simulation(N, d, RX, clusters_quantile, cluster_channels,smart_dynamic_select, arrivals, dev_sequence)

            reset_channel_stats()
            # ML Cluster Assignment
            ml_clusters = assign_clusters_ml(ml_model, d, RX, C)
            prob_ml = run_simulation(N, d, RX,ml_clusters,cluster_channels,random_select,arrivals,dev_sequence)
            
            # ----------------------------------------------------
            # PRINT RESULTS
            # ----------------------------------------------------
            if C == 8:
                print(
                    f"C={C}: "
                    f"[RandFull]={prob_rand_full:.4f}, "
                    f"[RandCh]={prob_random:.4f}, "
                    f"[LongTerm]={prob_smart_long_term:.4f}, "
                    f"[ML]={prob_ml:.4f}, "
                    f"Theory={theory:.4f}"
                )
            else:
                print(
                    f"C={C}: "
                    f"[RandFull]={prob_rand_full:.4f}, "
                    f"[RandCh]={prob_random:.4f}, "
                    f"[LongTerm]={prob_smart_long_term:.4f}, "
                    f"[ML]={prob_ml:.4f}"
                )


    plot_success_by_cluster_size(results)
    plot_success_by_device_count(results)
    plot_success_cdf()