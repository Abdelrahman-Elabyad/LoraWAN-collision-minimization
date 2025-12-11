# ============================================================
# visualize_clusters_by_power_2perfig.py
# Always plot 2 clusters per figure, colored by RX.
# Supports any C (C=1, C=2, C=4)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_FILE = "GA_dataset_master.csv"

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found.")

df = pd.read_csv(CSV_FILE)

required_cols = ["network_id", "x", "y", "RX", "cluster_optimal"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

print("Dataset loaded.")
print(f"Networks: {df['network_id'].nunique()}")

# ------------------------------------------------------------
# SELECT NETWORK
# ------------------------------------------------------------
nets = sorted(df["network_id"].unique())
print("\nAvailable network_ids:", nets)

while True:
    try:
        net_id = int(input("\nEnter network_id to visualize: "))
        if net_id in nets:
            break
    except:
        pass
    print("Invalid ID. Try again.")

df_net = df[df["network_id"] == net_id]
clusters = sorted(df_net["cluster_optimal"].unique())
K = len(clusters)

print(f"\nNetwork {net_id}: {len(df_net)} devices, {K} clusters.")

# ------------------------------------------------------------
# FUNCTION: always show 2 clusters per figure
# ------------------------------------------------------------
def plot_clusters_two_per_fig(df_net, clusters):
    """
    Always show 2 clusters per figure.
    If clusters = [0,1,2,3,4,5,6,7] → 4 figures.
    """
    clusters = list(clusters)
    total = len(clusters)
    num_figs = int(np.ceil(total / 2))

    for fig_idx in range(num_figs):
        start = fig_idx * 2
        end = min(start + 2, total)
        sub_clusters = clusters[start:end]

        fig, axes = plt.subplots(
            nrows=len(sub_clusters),
            ncols=1,
            figsize=(8, 6 * len(sub_clusters))
        )

        if len(sub_clusters) == 1:
            axes = [axes]  # Make iterable for single case

        for ax, cid in zip(axes, sub_clusters):
            df_c = df_net[df_net["cluster_optimal"] == cid]

            sc = ax.scatter(
                df_c["x"],
                df_c["y"],
                c=df_c["RX"],
                cmap="viridis",
                s=25,
                alpha=0.9
            )

            ax.set_title(f"Cluster {cid} — Colored by Power (RX)")
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")
            ax.grid(True)
            ax.axis("equal")

            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("RX Power (dBm)")

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plot_clusters_two_per_fig(df_net, clusters)

print("\nDone! Produced clean figures (2 clusters per figure).")
