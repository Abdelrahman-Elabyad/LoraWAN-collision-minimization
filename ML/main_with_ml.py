import numpy as np
import math
from clustering_channel_assignment import assign_clusters_quantile_stratified,assign_clusters_random, thompson_select_wrapper, random_select,scoring_select, smart_dynamic_select
from genetic_cluster_optimizer import build_cluster_channels
from simulation import run_simulation, generate_devices, reset_channel_stats, generate_arrivals
from simulation import plot_success_by_cluster_size, plot_success_by_device_count, plot_success_cdf
from simulation import PACKET_DURATION, MEAN_INTERARRIVAL, NUM_CHANNELS
from ML.cluster_ml_model import load_trained_model, assign_clusters_ml
from ML.cluster_rf_model import load_trained_rf, assign_clusters_rf

def aloha_theory_for_C(N, C):
    K = NUM_CHANNELS // C             # number of clusters
    N_cluster = N / K                 # avg devices per cluster
    lambda_cluster = N_cluster / MEAN_INTERARRIVAL     # arrivals/sec for cluster
    lambda_per_ch = lambda_cluster / C                 # arrivals/sec per channel
    G = lambda_per_ch * PACKET_DURATION
    return math.exp(-2 * G)

# ============================================================
# NEW: 20-RUN AVERAGING FUNCTION
# ============================================================
def avg_run(N, d, RX, clusters, cluster_channels, selector, arrivals, dev_seq, runs=100):
    total = 0.0
    for _ in range(runs):
        reset_channel_stats()
        total += run_simulation(N, d, RX, clusters, cluster_channels, selector, arrivals, dev_seq)
    return total / runs

if __name__ == "__main__":

    device_counts = [60000]
    cluster_sizes = [1]
    results = {}

    ml_model = load_trained_model("cluster_model_lgbm.bin")
    print("[INFO] ML model loaded.")

    for N in device_counts:
        print(f"\n========== N = {N} ==========")
        d, RX, positions = generate_devices(N)
        # SAME arrivals for all algorithms
        arrivals = generate_arrivals(N)
        dev_sequence = np.random.randint(0, N, size=len(arrivals))
        
        for C in cluster_sizes:
            cluster_channels, _ = build_cluster_channels(C)
            theory = aloha_theory_for_C(N, C)
            # ----------------------------------------------------
            # 1) TRUE RANDOM BASELINE (Random clusters + Random channels)
            # ----------------------------------------------------
            # 1) TRUE RANDOM BASELINE
            clusters_rand = assign_clusters_random(N, C)
            prob_rand_full = avg_run(N, d, RX, clusters_rand, cluster_channels,
                                    random_select, arrivals, dev_sequence, runs=20)

            # 2) SMART STATIC BASELINE
            clusters_quantile = assign_clusters_quantile_stratified(d, RX, C)

            prob_random = avg_run(N, d, RX, clusters_quantile, cluster_channels,
                                random_select, arrivals, dev_sequence, runs=20)

            prob_smart_long_term = avg_run(N, d, RX, clusters_quantile, cluster_channels,
                                        smart_dynamic_select, arrivals, dev_sequence, runs=20)

            # 3) ML CLUSTER ASSIGNMENT
            #ml_clusters = assign_clusters_ml(ml_model, d, RX, C, positions)
            #prob_ml = avg_run(N, d, RX, ml_clusters, cluster_channels,random_select, arrivals, dev_sequence, runs=20)
            
            # 4) RF CLUSTER ASSIGNMENT
            # Load Random Forest
            rf_model = load_trained_rf("cluster_model_rf.bin")

            # ML cluster assignment using Random Forest
            clusters_rf = assign_clusters_rf(rf_model, d, RX, C, positions)
            prob_rf = avg_run(N, d, RX, clusters_rf, cluster_channels,random_select, arrivals, dev_sequence, runs=20)

            # ----------------------------------------------------
            # PRINT RESULTS
            # ----------------------------------------------------
            print(
            f"C={C}: "
            f"[RandFull]={prob_rand_full:.4f}, "
            f"[RandCh]={prob_random:.4f}, "
            f"[LongTerm]={prob_smart_long_term:.4f}, "
            f"[RF]={prob_rf:.4f}"
            f"Theory(C={C})={theory:.4f}"
            )

    '''
    plot_success_by_cluster_size(results)
    plot_success_by_device_count(results)
    plot_success_cdf()
    '''