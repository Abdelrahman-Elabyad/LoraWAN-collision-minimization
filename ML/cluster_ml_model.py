import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


# ============================================================
# 1) FEATURE EXTRACTION (NEW, STABLE, NEIGHBOR-AWARE)
# ============================================================

def extract_features(d, RX, C, positions, R=80, decay=120):
    """
    Stable, generalizable features for ML cluster assignment.

    d: distance array
    RX: RSSI values
    C: cluster size (channels per cluster)
    positions: Nx2 array of device coordinates
    """

    N = len(d)
    K = 8 // C   # number of clusters

    # -------------------------------------------
    # 1. Rank-bin features (0–9)
    # -------------------------------------------
    d_rank = np.argsort(np.argsort(d)) / N
    rx_rank = np.argsort(np.argsort(RX)) / N

    d_bin = (d_rank * 10).astype(int)
    rx_bin = (rx_rank * 10).astype(int)

    # -------------------------------------------
    # 2. Local neighborhood statistics
    # -------------------------------------------
    local_density = np.zeros(N)
    local_rx_var = np.zeros(N)
    interference = np.zeros(N)

    AREA = np.pi * (1000**2)     # coverage area estimate
    expected_density = N / AREA  # normalization factor

    Xpos = positions[:, 0]
    Ypos = positions[:, 1]

    for i in range(N):
        dx = Xpos - Xpos[i]
        dy = Ypos - Ypos[i]
        dist = np.sqrt(dx*dx + dy*dy)

        mask = (dist < R) & (dist > 0)
        neigh_rx = RX[mask]

        local_density[i] = len(neigh_rx) / max(1e-9, expected_density)
        local_rx_var[i] = np.var(neigh_rx) if len(neigh_rx) > 1 else 0
        interference[i] = np.sum(np.exp(-dist[mask] / decay) * neigh_rx)

    # -------------------------------------------
    # 3. Sector ID (angular region 0–7)
    # -------------------------------------------
    angles = (np.arctan2(Ypos, Xpos) + np.pi) / (2 * np.pi)
    sector_id = (angles * 8).astype(int)

    # -------------------------------------------
    # 4. Expected load (scale-invariant)
    # -------------------------------------------
    expected_load = np.full(N, N / K)

    # -------------------------------------------
    # 5. Build the complete feature matrix
    # -------------------------------------------
    X = np.column_stack([
        d, RX, np.full(N, C),
        d_bin, rx_bin,
        local_density, local_rx_var,
        sector_id, expected_load,
        interference
    ])

    return X


# ============================================================
# 2) LOAD TRAINING DATA
# ============================================================

def load_training_data(csv_path="GA_dataset_master.csv"):
    print(f"[ML] Loading dataset from {csv_path} ...")

    df = pd.read_csv(csv_path)

    required = ["N", "device_id", "d", "RX", "C",
                "cluster_optimal", "x", "y"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column: {col}")

    return df


# ============================================================
# 3) TRAIN THE MODEL (LIGHTGBM MULTICLASS)
# ============================================================

def train_lightgbm_model(df, model_path="cluster_model_lgbm.bin"):
    print("[ML] Extracting features...")

    d = df["d"].to_numpy()
    RX = df["RX"].to_numpy()
    C = df["C"].iloc[0]               # cluster size is fixed per GA dataset
    pos = df[["x", "y"]].to_numpy()

    X = extract_features(d, RX, C, pos)
    y = df["cluster_optimal"].to_numpy()

    print(f"[ML] Feature matrix shape: {X.shape}")

    train_data = lgb.Dataset(X, label=y)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "metric": "multi_logloss",
        "verbosity": -1
    }

    print("[ML] Training LightGBM...")
    model = lgb.train(params, train_data, num_boost_round=1200)

    joblib.dump(model, model_path)
    print(f"[ML] Saved model → {model_path}")

    return model


# ============================================================
# 4) LOAD TRAINED MODEL
# ============================================================

def load_trained_model(model_path="cluster_model_lgbm.bin"):
    print("[ML] Loading trained model...")
    return joblib.load(model_path)


# ============================================================
# 5) ASSIGN CLUSTERS USING ML MODEL
# ============================================================

def assign_clusters_ml(model, d, RX, C, positions):
    X = extract_features(d, RX, C, positions)
    preds = model.predict(X)
    clusters = preds.argmax(axis=1)

    K = 8 // C
    return np.clip(clusters, 0, K - 1)


# ============================================================
# 6) STANDALONE TRAINING ENTRY POINT
# ============================================================

if __name__ == "__main__":
    df = load_training_data("GA_dataset_master.csv")
    model = train_lightgbm_model(df, "cluster_model_lgbm.bin")
    print("\n[ML] Training completed successfully!")
