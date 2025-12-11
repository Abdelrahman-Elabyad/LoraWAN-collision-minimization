# ============================================================
# visualize_clusters_side_by_side_save.py
# Guaranteed-working version: saves PNG files correctly.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_COMPARE = "cluster_compare_runs.csv"
CSV_GEOMETRY = "GA_dataset_master.csv"

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------
df_clusters = pd.read_csv(CSV_COMPARE)
df_geo = pd.read_csv(CSV_GEOMETRY)

# ------------------------------------------------------------
# Ask user which run to visualize
# ------------------------------------------------------------
run_ids = sorted(df_clusters["run_id"].unique())
print("\nAvailable run_ids:", run_ids)

while True:
    try:
        chosen_run = int(input("\nEnter run_id: "))
        if chosen_run in run_ids:
            break
    except:
        pass
    print("Invalid run_id, try again.")

df_run = df_clusters[df_clusters["run_id"] == chosen_run].sort_values("device_id")

N = df_run["N"].iloc[0]
C = df_run["C"].iloc[0]

df_geo_run = df_geo[df_geo["N"] == N].sort_values("device_id")

x = df_geo_run["x"].values
y = df_geo_run["y"].values
RX = df_geo_run["RX"].values

clusters_random = df_run["cluster_random"].values
clusters_quantile = df_run["cluster_quantile"].values
clusters_ga = df_run["cluster_GA"].values

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
save_dir = f"plots_run_{chosen_run}"
os.makedirs(save_dir, exist_ok=True)
print(f"\nSaving all PNGs into: {os.path.abspath(save_dir)}\n")

# ------------------------------------------------------------
# Helper: plot 2 clusters per figure and save
# ------------------------------------------------------------
def plot_two_per_fig(x, y, RX, clusters, method_name, run_id):

    unique_clusters = sorted(np.unique(clusters))
    total = len(unique_clusters)
    num_figures = int(np.ceil(total / 2))

    for fig_idx in range(num_figures):

        start = fig_idx * 2
        end = min(start + 2, total)
        subclust = unique_clusters[start:end]

        fig, axes = plt.subplots(1, len(subclust), figsize=(12, 6))

        if len(subclust) == 1:
            axes = [axes]  # make iterable

        for ax, cid in zip(axes, subclust):

            mask = clusters == cid
            sc = ax.scatter(
                x[mask], y[mask],
                c=RX[mask],
                cmap="viridis",
                s=18,
                alpha=0.9
            )

            ax.set_title(f"{method_name} — Cluster {cid}")
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.grid(True)
            ax.axis("equal")

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("RX Power (dBm)")

        plt.tight_layout()

        # ---------- Safe filename ----------
        clusters_id_str = "_".join(str(c) for c in subclust)
        filename = f"{method_name.lower().replace(' ', '_')}_clusters_{clusters_id_str}.png"
        filepath = os.path.join(save_dir, filename)

        # ---------- SAVE BEFORE SHOW ----------
        plt.savefig(filepath, dpi=300)
        print(f"[Saved] {filepath}")

        plt.show()


# ------------------------------------------------------------
# Generate visualizations
# ------------------------------------------------------------
print(f"\nGenerating plots for run {chosen_run}...\n")

plot_two_per_fig(x, y, RX, clusters_random, "Random_Baseline", chosen_run)
plot_two_per_fig(x, y, RX, clusters_quantile, "Quantile_Clustering", chosen_run)
plot_two_per_fig(x, y, RX, clusters_ga, "Genetic_Algorithm", chosen_run)

print("\nAll PNG files saved successfully!")
