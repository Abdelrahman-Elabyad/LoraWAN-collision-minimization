import os
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import math
from genetic_cluster_optimizer import optimize_clusters_with_GA, USE_MULTIPROCESSING, build_cluster_channels
from simulation import run_simulation, generate_devices, generate_arrivals, reset_channel_stats, compute_cdf
from clustering_channel_assignment import assign_clusters_random, assign_clusters_quantile_stratified, random_select
from simulation import NUM_CHANNELS, PACKET_DURATION, MEAN_INTERARRIVAL

RUN_PLAN = {
    200: 1,
    1000: 1,
    6000: 1,
    #10000: 1,
    20000: 1,
    46000: 1,
    #80000: 1,
    #100000: 1
}

CLUSTER_SIZES = [1,2,4]


def aloha_theory_for_C(N, C):
    """Calculate theoretical Slotted ALOHA success probability for given N and C."""
    K = NUM_CHANNELS // C             # number of clusters
    N_cluster = N / K                 # avg devices per cluster
    lambda_cluster = N_cluster / MEAN_INTERARRIVAL     # arrivals/sec for cluster
    lambda_per_ch = lambda_cluster / C                 # arrivals/sec per channel
    G = lambda_per_ch * PACKET_DURATION
    return math.exp(-2 * G)


def save_cdf_plot(xs, ys, name):
    """Save CDF plot to file in cdf_plots directory."""
    if xs is None or len(xs) == 0:
        print(f"[WARNING] No CDF data available for {name}. Skipping plot.")
        return

    # Create cdf_plots directory if it doesn't exist
    plots_dir = "cdf_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"[INFO] Created directory: {plots_dir}")

    print(f"[PLOT] Saving CDF plot: {name}.png ({len(xs)} points)")

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, linewidth=2, color='blue', alpha=0.7)
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("CDF of Successful Packets", fontsize=12)
    plt.title(name.replace("_", " "), fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(plots_dir, name + ".png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {os.path.abspath(output_path)}")


def save_comparison_cdf_plot(data_dict, N, C):
    """
    Save a comparison CDF plot with Quantile and GA methods in cdf_plots directory.
    
    Args:
        data_dict: Dictionary with keys 'Quantile', 'GA'
                   each containing (xs, ys) tuples
    """
    # Create cdf_plots directory if it doesn't exist
    plots_dir = "cdf_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"[INFO] Created directory: {plots_dir}")
    
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Quantile': 'orange',
        'GA': 'green'
    }
    
    labels = {
        'Quantile': 'Quantile Stratified Clusters',
        'GA': 'GA-Optimized Clusters'
    }
    
    for method, (xs, ys) in data_dict.items():
        if xs is not None and len(xs) > 0:
            plt.plot(xs, ys, linewidth=2.5, alpha=0.8, 
                    color=colors[method], label=labels[method])
    
    plt.xlabel("Distance from Gateway (km)", fontsize=12)
    plt.ylabel("CDF of Successful Packets", fontsize=12)
    plt.title(f"Success Distance CDF Comparison (N={N}, C={C})", 
             fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(plots_dir, f"CDF_Comparison_N{N}_C{C}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED COMPARISON] {os.path.abspath(output_path)}")


def format_time(seconds):
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m ({seconds:.2f}s)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.2f}h ({minutes:.1f}m)"


def log_runtime_to_csv(run_id, N, C, ga_time, total_time, csv_file="runtime_log.csv"):
    """Log runtime statistics to CSV."""
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "run_id", "N", "C", "ga_time_seconds", "total_time_seconds",
                "ga_time_formatted", "total_time_formatted"
            ])
        
        writer.writerow([
            run_id, N, C, f"{ga_time:.2f}", f"{total_time:.2f}",
            format_time(ga_time), format_time(total_time)
        ])
    
    print(f"[RUNTIME] Logged to {csv_file}")


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    print("\n========== GA OPTIMIZATION STARTED ==========")
    print(f"Multiprocessing Enabled: {USE_MULTIPROCESSING}")
    print("---------------------------------------------------\n")

    # Start overall timer
    overall_start_time = time.time()
    
    run_id = 0
    all_runtimes = []

    for N, repetitions in RUN_PLAN.items():
        for C in CLUSTER_SIZES:
            for rep in range(repetitions):
                run_id += 1
                
                # Start timer for this specific run
                run_start_time = time.time()
                
                print(f"\n>>> GA RUN {run_id}: N={N}, C={C}, repetition {rep+1}/{repetitions}")
                print("---------------------------------------------------")

                # =============================
                # THEORETICAL LIMIT
                # =============================
                theoretical_limit = aloha_theory_for_C(N, C)
                print(f"[THEORY] Slotted ALOHA theoretical limit: {theoretical_limit:.4f}")
                print("---------------------------------------------------")

                # =============================
                # GENERATE DEVICES + ARRIVALS
                # =============================
                print("[INFO] Generating devices (geometry) and arrivals...")
                d, RX, positions = generate_devices(N)
                arrivals = generate_arrivals(N)
                dev_sequence = np.random.randint(0, N, size=len(arrivals))

                cluster_channels = build_cluster_channels(C)[0]

                # =============================
                # BASELINE: RandFull (no CDF needed)
                # =============================
                print("[INFO] Running RandFull baseline...")
                baseline_start = time.time()
                clusters_rand = np.random.randint(0, 8//C, size=N)
                reset_channel_stats()
                randfull_score, _, _ = run_simulation(
                    N, d, RX, clusters_rand, cluster_channels, random_select, arrivals, dev_sequence
                )
                baseline_time = time.time() - baseline_start
                print(f"[INFO] RandFull score: {randfull_score:.4f} (time: {baseline_time:.2f}s)")

                # =============================
                # BASELINE: Quantile (WITH CDF)
                # =============================
                print("[INFO] Running Quantile baseline...")
                quantile_start = time.time()
                clusters_quant = assign_clusters_quantile_stratified(d, RX, C)
                reset_channel_stats()
                quantile_score, succ_quant, fail_quant = run_simulation(
                    N, d, RX, clusters_quant, cluster_channels, random_select, arrivals, dev_sequence
                )
                quantile_time = time.time() - quantile_start
                print(f"[INFO] Quantile score: {quantile_score:.4f} (time: {quantile_time:.2f}s)")

                xs_quant, ys_quant = compute_cdf(succ_quant)
                save_cdf_plot(xs_quant, ys_quant, f"CDF_Quantile_N{N}_C{C}")

                # =============================
                # RUN GA (includes final evaluation with CDF)
                # =============================
                print("[INFO] Running Genetic Algorithm...")
                print("⏱️  [TIMER] GA optimization started...")
                ga_start_time = time.time()
                
                best_genome, ga_score, succ_ga, fail_ga = optimize_clusters_with_GA(
                    N=N,
                    C=C,
                    d=d,
                    RX=RX,
                    positions=positions,
                    arrivals=arrivals,
                    dev_sequence=dev_sequence
                )
                
                ga_elapsed_time = time.time() - ga_start_time
                print(f"⏱️  [TIMER] GA optimization completed in {format_time(ga_elapsed_time)}")
                
                print(f"[INFO] GA returned: score={ga_score:.4f}, "
                      f"success_count={len(succ_ga) if succ_ga else 0}, "
                      f"failure_count={len(fail_ga) if fail_ga else 0}")
                
                # Safety check
                if succ_ga is None:
                    print("[WARNING] GA did not return success distances, using empty list")
                    succ_ga = []
                if fail_ga is None:
                    print("[WARNING] GA did not return failure distances, using empty list")
                    fail_ga = []

                xs_ga, ys_ga = compute_cdf(succ_ga)
                save_cdf_plot(xs_ga, ys_ga, f"CDF_GA_N{N}_C{C}")

                # =============================
                # SAVE COMPARISON PLOT (Quantile vs GA only)
                # =============================
                print("[INFO] Creating comparison plot (Quantile vs GA)...")
                comparison_data = {
                    'Quantile': (xs_quant, ys_quant),
                    'GA': (xs_ga, ys_ga)
                }
                save_comparison_cdf_plot(comparison_data, N, C)

                # Calculate total run time
                run_elapsed_time = time.time() - run_start_time
                
                # Store runtime data
                all_runtimes.append({
                    'run_id': run_id,
                    'N': N,
                    'C': C,
                    'ga_time': ga_elapsed_time,
                    'total_time': run_elapsed_time
                })
                
                # Log runtime to CSV
                log_runtime_to_csv(run_id, N, C, ga_elapsed_time, run_elapsed_time)

                # =============================
                # DETAILED STATISTICS
                # =============================
                print("\n============= DETAILED STATISTICS =============")
                print(f"N={N}, C={C}")
                print("\n--- Theoretical Limit ---")
                print(f"ALOHA Theory: {theoretical_limit:.4f}")
                
                print("\n--- Success Probabilities ---")
                print(f"RandFull:  {randfull_score:.4f}  (Theory: {theoretical_limit:.4f}, Ratio: {randfull_score/theoretical_limit:.3f})")
                print(f"Quantile:  {quantile_score:.4f}  (Theory: {theoretical_limit:.4f}, Ratio: {quantile_score/theoretical_limit:.3f})")
                print(f"GA:        {ga_score:.4f}  (Theory: {theoretical_limit:.4f}, Ratio: {ga_score/theoretical_limit:.3f})")
                
                print("\n--- Improvements ---")
                improvement_rand = ga_score - randfull_score
                improvement_quant = ga_score - quantile_score
                improvement_theory = ga_score - theoretical_limit
                pct_rand = (ga_score/randfull_score - 1)*100 if randfull_score > 0 else 0
                pct_quant = (ga_score/quantile_score - 1)*100 if quantile_score > 0 else 0
                pct_theory = (ga_score/theoretical_limit - 1)*100 if theoretical_limit > 0 else 0
                
                print(f"GA vs RandFull:  {improvement_rand:+.4f}  ({pct_rand:+.2f}%)")
                print(f"GA vs Quantile:  {improvement_quant:+.4f}  ({pct_quant:+.2f}%)")
                print(f"GA vs Theory:    {improvement_theory:+.4f}  ({pct_theory:+.2f}%)")
                
                print("\n--- Success/Failure Counts ---")
                print(f"RandFull:  (not tracked)")
                print(f"Quantile:  {len(succ_quant):5d} success, {len(fail_quant):5d} failures")
                print(f"GA:        {len(succ_ga):5d} success, {len(fail_ga):5d} failures")
                
                if len(succ_ga) > 0 and len(succ_quant) > 0:
                    print("\n--- Mean Success Distance ---")
                    print(f"Quantile:  {np.mean(succ_quant):.4f} km")
                    print(f"GA:        {np.mean(succ_ga):.4f} km")
                    
                    print("\n--- Median Success Distance ---")
                    print(f"Quantile:  {np.median(succ_quant):.4f} km")
                    print(f"GA:        {np.median(succ_ga):.4f} km")
                
                print("\n--- RUNTIME SUMMARY ---")
                print(f"Baseline (RandFull):  {format_time(baseline_time)}")
                print(f"Baseline (Quantile):  {format_time(quantile_time)}")
                print(f"GA Optimization:      {format_time(ga_elapsed_time)}")
                print(f"Total Run Time:       {format_time(run_elapsed_time)}")
                print("===============================================\n")

    # Calculate overall statistics
    overall_elapsed_time = time.time() - overall_start_time
    
    print("\n========== FINAL RUNTIME REPORT ==========")
    print(f"Total Runs Completed: {run_id}")
    print(f"Overall Runtime: {format_time(overall_elapsed_time)}")
    print("\n--- Per-Run Breakdown ---")
    
    for runtime_data in all_runtimes:
        print(f"Run {runtime_data['run_id']}: N={runtime_data['N']}, C={runtime_data['C']} | "
              f"GA: {format_time(runtime_data['ga_time'])} | "
              f"Total: {format_time(runtime_data['total_time'])}")
    
    # Calculate averages by C
    print("\n--- Average Times by Cluster Size ---")
    for C in CLUSTER_SIZES:
        c_runtimes = [r for r in all_runtimes if r['C'] == C]
        if c_runtimes:
            avg_ga = np.mean([r['ga_time'] for r in c_runtimes])
            avg_total = np.mean([r['total_time'] for r in c_runtimes])
            print(f"C={C}: GA={format_time(avg_ga)}, Total={format_time(avg_total)}")
    
    print("\n========== GA OPTIMIZATION COMPLETED ==========\n")