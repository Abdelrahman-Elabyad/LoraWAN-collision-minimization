# ============================================================
# visualize_clusters.py
# Visualize clusters in GA_dataset_master.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
CSV_FILE = "GA_dataset_master.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found in current directory.")

df = pd.read_csv(CSV_FILE)

required_cols = ["network_id", "x", "y", "d", "RX", "cluster_optimal"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column missing in CSV: {col}")

print("====================================================")
print(" GA DATASET LOADED")
print("====================================================")
print(f"Total networks: {df['network_id'].nunique()}")
print(f"Total devices:  {len(df)}")
print("====================================================")

# ------------------------------------------------------------
# Ask user to choose a network
# ------------------------------------------------------------
unique_networks = sorted(df["network_id"].unique())

print("\nAvailable Network IDs:")
print(unique_networks)

while True:
    try:
        net_id = int(input("\nEnter network_id to visualize: "))
        if net_id in unique_networks:
            break
        else:
            print("Invalid network_id. Try again.")
    except:
        print("Please enter a valid integer.")

# Filter dataset
df_net = df[df["network_id"] == net_id].copy()

clusters = df_net["cluster_optimal"].values
x = df_net["x"].values
y = df_net["y"].values

K = len(df_net["cluster_optimal"].unique())

print(f"\nSelected Network {net_id} contains {len(df_net)} devices and {K} clusters.")

# ------------------------------------------------------------
# Compute centroids
# ------------------------------------------------------------
centroids = df_net.groupby("cluster_optimal")[["x", "y"]].mean()

# ------------------------------------------------------------
# VISUALIZATION 1: XY SCATTER PLOT
# ------------------------------------------------------------
plt.figure(figsize=(10, 10))
colors = cm.tab20(np.linspace(0, 1, K))

for idx, cluster_id in enumerate(sorted(df_net["cluster_optimal"].unique())):
    mask = df_net["cluster_optimal"] == cluster_id
    plt.scatter(
        df_net[mask]["x"],
        df_net[mask]["y"],
        color=colors[idx],
        label=f"Cluster {cluster_id}",
        s=20,
        alpha=0.6
    )

# Plot centroids
for idx, cluster_id in enumerate(sorted(df_net["cluster_optimal"].unique())):
    cx, cy = centroids.loc[cluster_id]
    plt.scatter(cx, cy, color=colors[idx], edgecolor='black',
                s=200, marker='X', linewidth=1.5)

plt.title(f"Network {net_id} — Cluster Visualization", fontsize=16)
plt.xlabel("X coordinate (km)")
plt.ylabel("Y coordinate (km)")
plt.legend(loc="upper right")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# VISUALIZATION 2: DISTANCE HISTOGRAM PER CLUSTER
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

max_dist = df_net["d"].max()

for idx, cluster_id in enumerate(sorted(df_net["cluster_optimal"].unique())):
    mask = df_net["cluster_optimal"] == cluster_id
    plt.hist(
        df_net[mask]["d"],
        bins=30,
        alpha=0.5,
        color=colors[idx],
        label=f"Cluster {cluster_id}"
    )

plt.title(f"Network {net_id} — Distance Distribution per Cluster", fontsize=16)
plt.xlabel("Distance from Gateway (km)")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# END
# ------------------------------------------------------------
print("\nVisualization completed successfully!")
