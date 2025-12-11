import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# Load dataset with network_id
# ============================================================
def load_training_data(csv_path="GA_dataset_master.csv"):
    df = pd.read_csv(csv_path)
    
    required = ["network_id", "N", "device_id", "d", "RX", "C", "cluster_optimal", "x", "y"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in dataset: {col}")
    
    return df


# ============================================================
# ENHANCED Feature Extraction (Best of Both Worlds)
# ============================================================
def extract_network_aware_features(df_network, R=80, decay=120):
    """
    Enhanced feature extraction combining:
    - Your original sophisticated local features (interference, sectors, etc.)
    - Network-aware contextual features (ranks, density, global stats)
    
    Returns enriched feature matrix with full network context.
    """
    N = len(df_network)
    
    d = df_network["d"].values
    RX = df_network["RX"].values
    x = df_network["x"].values
    y = df_network["y"].values
    C = df_network["C"].iloc[0]
    K = 8 // C  # number of clusters
    
    positions = np.column_stack([x, y])
    
    # ============================================================
    # PART 1: YOUR ORIGINAL FEATURES (Proven & Sophisticated)
    # ============================================================
    
    # 1. Rank-bin features (0–9)
    d_rank = np.argsort(np.argsort(d)) / N
    rx_rank = np.argsort(np.argsort(RX)) / N
    
    d_bin = (d_rank * 10).astype(int)
    rx_bin = (rx_rank * 10).astype(int)
    
    # 2. Local neighborhood statistics
    local_density = np.zeros(N)
    local_rx_var = np.zeros(N)
    interference = np.zeros(N)
    
    AREA = np.pi * (1000**2)
    expected_density = N / AREA
    
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        
        mask = (dist < R) & (dist > 0)
        neigh_rx = RX[mask]
        
        local_density[i] = len(neigh_rx) / max(1e-9, expected_density)
        local_rx_var[i] = np.var(neigh_rx) if len(neigh_rx) > 1 else 0
        interference[i] = np.sum(np.exp(-dist[mask] / decay) * neigh_rx)
    
    # 3. Sector ID (angular region 0–7)
    angles = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    sector_id = (angles * 8).astype(int)
    
    # 4. Expected load (scale-invariant)
    expected_load = np.full(N, N / K)
    
    # ============================================================
    # PART 2: NETWORK-AWARE ENHANCEMENTS
    # ============================================================
    
    # 5. Multi-scale neighborhood counts (different radii)
    neighbors_50m = np.zeros(N)
    neighbors_100m = np.zeros(N)
    neighbors_200m = np.zeros(N)
    
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        
        neighbors_50m[i] = np.sum((dist < 50) & (dist > 0))
        neighbors_100m[i] = np.sum((dist < 100) & (dist > 0))
        neighbors_200m[i] = np.sum((dist < 200) & (dist > 0))
    
    # 6. Distance to network centroid
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    dist_to_centroid = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
    
    # 7. Statistical deviation features
    d_deviation = (d - np.mean(d)) / (np.std(d) + 1e-9)
    RX_deviation = (RX - np.mean(RX)) / (np.std(RX) + 1e-9)
    
    # 8. Nearest neighbor distance
    nearest_neighbor_dist = np.zeros(N)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        dist[i] = np.inf  # exclude self
        nearest_neighbor_dist[i] = np.min(dist)
    
    # 9. Radial position (distance from center, normalized)
    radial_position = dist_to_centroid / (np.max(dist_to_centroid) + 1e-9)
    
    # 10. Local RX statistics (average of nearby devices)
    avg_nearby_RX = np.zeros(N)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        close_mask = (dist < 150) & (dist > 0)
        if np.sum(close_mask) > 0:
            avg_nearby_RX[i] = np.mean(RX[close_mask])
        else:
            avg_nearby_RX[i] = RX[i]
    
    # 11. Network-level aggregate features (same for all devices in network)
    network_mean_d = np.full(N, np.mean(d))
    network_std_d = np.full(N, np.std(d))
    network_mean_RX = np.full(N, np.mean(RX))
    network_std_RX = np.full(N, np.std(RX))
    network_size = np.full(N, N)
    
    # 12. Density gradient (change in density with radius)
    density_gradient = np.zeros(N)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        
        inner = np.sum((dist < 100) & (dist > 0))
        outer = np.sum((dist >= 100) & (dist < 200))
        
        density_gradient[i] = (inner - outer) / max(inner + outer, 1)
    
    # ============================================================
    # ASSEMBLE COMPLETE FEATURE MATRIX
    # ============================================================
    X = np.column_stack([
        # Original features (10 features)
        d, RX, np.full(N, C),
        d_bin, rx_bin,
        local_density, local_rx_var,
        sector_id, expected_load,
        interference,
        
        # Network-aware enhancements (13 features)
        d_rank, rx_rank,  # Continuous ranks (more info than bins)
        neighbors_50m, neighbors_100m, neighbors_200m,
        dist_to_centroid, radial_position,
        d_deviation, RX_deviation,
        nearest_neighbor_dist,
        avg_nearby_RX,
        density_gradient,
        
        # Network-level context (5 features)
        network_size, network_mean_d, network_std_d,
        network_mean_RX, network_std_RX
    ])
    
    return X


# ============================================================
# Load and prepare features for entire dataset (network-aware)
# ============================================================
def load_and_extract_features(csv_path="GA_dataset_master.csv"):
    """
    Load dataset and extract features network by network.
    Returns X, y, and groups (network_id for each sample).
    """
    df = load_training_data(csv_path)
    
    all_features = []
    all_labels = []
    all_groups = []
    
    # Process each network separately
    for network_id in df["network_id"].unique():
        df_network = df[df["network_id"] == network_id].sort_values("device_id")
        
        # Extract features for this network
        X_network = extract_network_aware_features(df_network)
        y_network = df_network["cluster_optimal"].values
        
        all_features.append(X_network)
        all_labels.append(y_network)
        all_groups.extend([network_id] * len(df_network))
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.array(all_groups)
    
    return X, y, groups


# ============================================================
# Train Random Forest with Network Grouping
# ============================================================
def train_rf_model(csv_path="GA_dataset_master.csv", model_path="cluster_model_rf.bin"):
    print("[RF] Loading dataset and extracting network-aware features...")
    
    X, y, groups = load_and_extract_features(csv_path)
    
    print(f"[RF] Feature matrix shape: {X.shape}")
    print(f"[RF] Number of features: {X.shape[1]}")
    print(f"[RF] Number of unique networks: {len(np.unique(groups))}")
    print(f"[RF] Number of devices: {len(y)}")
    print(f"[RF] Number of clusters: {len(np.unique(y))}")
    
    model = RandomForestClassifier(
        n_estimators=500,  # Increased for richer features
        max_depth=30,      # Increased for complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Prevent overfitting with many features
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )
    
    print("[RF] Training RandomForest...")
    model.fit(X, y)
    
    # Save trained model
    joblib.dump(model, model_path)
    print(f"[RF] Saved model → {model_path}")
    
    return model


# ============================================================
# Evaluate with Network-Aware Split (NO DATA LEAKAGE)
# ============================================================
def evaluate_rf_model(csv_path="GA_dataset_master.csv", test_size=0.25, 
                      random_state=42, model_path="cluster_model_rf.bin"):
    """
    Train/evaluate RandomForest with NETWORK-GROUPED train/test split.
    This ensures entire networks are either in train OR test, never split.
    """
    print("[RF] Loading dataset and extracting network-aware features...")
    
    X, y, groups = load_and_extract_features(csv_path)
    
    print(f"[RF] Feature matrix shape: {X.shape}")
    print(f"[RF] Number of features: {X.shape[1]}")
    print(f"[RF] Number of unique networks: {len(np.unique(groups))}")
    
    # ============ CRITICAL: GROUP-AWARE SPLIT ============
    # Entire networks go to train or test, preventing data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    
    print(f"[RF] Train shape: {X_train.shape} ({len(np.unique(groups_train))} networks)")
    print(f"[RF] Test shape: {X_test.shape} ({len(np.unique(groups_test))} networks)")
    
    # Verify no overlap
    assert len(set(groups_train) & set(groups_test)) == 0, "ERROR: Network leakage detected!"
    print("[RF] ✓ No network leakage between train/test")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state
    )
    
    print("[RF] Training RandomForest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n[RF] Accuracy: {acc:.4f}")
    print("[RF] Classification Report:\n", report)
    print("[RF] Confusion Matrix:\n", cm)
    
    # Feature importance analysis
    feature_names = [
        # Original features
        "d", "RX", "C", "d_bin", "rx_bin",
        "local_density", "local_rx_var", "sector_id", 
        "expected_load", "interference",
        # Network-aware additions
        "d_rank", "rx_rank",
        "neighbors_50m", "neighbors_100m", "neighbors_200m",
        "dist_to_centroid", "radial_position",
        "d_deviation", "RX_deviation",
        "nearest_neighbor_dist", "avg_nearby_RX",
        "density_gradient",
        # Network context
        "network_size", "network_mean_d", "network_std_d",
        "network_mean_RX", "network_std_RX"
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    print("\n[RF] Top 15 Most Important Features:")
    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:22s}: {importances[idx]:.4f}")
    
    # Save evaluated model
    joblib.dump(model, model_path)
    print(f"\n[RF] Saved evaluated model → {model_path}")
    
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}, model


# ============================================================
# Prediction function for new networks
# ============================================================
def load_trained_rf(model_path="cluster_model_rf.bin"):
    """Load a trained Random Forest model."""
    return joblib.load(model_path)


def assign_clusters_rf(model, d, RX, C, positions):
    """
    Predict cluster assignments for a new network using trained RF model.
    
    Args:
        model: Trained RandomForest model
        d: distances (N,)
        RX: received power (N,)
        C: cluster size
        positions: (N, 2) array of [x, y] coordinates
    
    Returns:
        clusters: (N,) array of cluster assignments
    """
    N = len(d)
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Create temporary dataframe to use extract_network_aware_features
    df_network = pd.DataFrame({
        "d": d,
        "RX": RX,
        "x": x,
        "y": y,
        "C": [C] * N
    })
    
    # Extract features
    X = extract_network_aware_features(df_network)
    
    # Predict
    clusters = model.predict(X)
    
    return clusters


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("[RF] Loading dataset and running network-aware evaluation...")
    results, trained_model = evaluate_rf_model(
        csv_path="GA_dataset_master.csv",
        test_size=0.25,
        random_state=42,
        model_path="cluster_model_rf.bin"
    )
    
    print("\n[RF] Evaluation completed. Summary:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print("  Classification report:\n", results["report"])