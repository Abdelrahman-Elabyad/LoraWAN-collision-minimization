import numpy as np
import json
from simulation import channel_stats, USE_ML
import simulation 


# Recommended static parameters for maximum success (without ML)
# Old default static parameters
# alpha = 1.0
# beta = 0.5
# gamma = 1.5
# lambda_val = 1.0

alpha = 1.5
beta  = 0.2
gamma = 2.0
lambda_val = 3.0

# If ML enabled, load optimized parameters
if USE_ML:
    try:
        with open("best_params.json", "r") as f:
            params = json.load(f)
        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        lambda_val = params["lambda"]
    except:
        print("[WARNING] best_params.json not found → using default parameters.")

# ----------------------------------------------------
# STATIC CLUSTER ASSIGNMENT BASED ON DISTANCE/RX
# ----------------------------------------------------
def assign_clusters_quantile_stratified(d, RX, C, NUM_CHANNELS=8):
    """
    Power-quantile stratified static cluster assignment with dynamic bucket count.
    Q is chosen based on N = number of devices.

    Steps:
    - Sort by RX (desc)
    - Split into Q quantile buckets
    - Round-robin assign buckets into clusters
    """

    N = len(d)
    K = NUM_CHANNELS // C      # number of clusters

    # -------------------------------
    # Dynamic quantile count Q(N)
    # -------------------------------
    Q = int(round(np.sqrt(N) / 5))
    Q = max(4, min(Q, 30))     # keep Q between 4 and 30

    clusters = np.zeros(N, dtype=int)

    # 1. Sort devices by RX descending
    sorted_idx = np.argsort(-RX)

    # 2. Build quantile buckets
    buckets = []
    bucket_size = int(np.ceil(N / Q))

    for q in range(Q):
        start = q * bucket_size
        end = min((q + 1) * bucket_size, N)
        if start < end:
            buckets.append(sorted_idx[start:end])

    # 3. Round-robin assignment across clusters
    cluster_id = 0
    for bucket in buckets:
        for dev_idx in bucket:
            clusters[dev_idx] = cluster_id
            cluster_id = (cluster_id + 1) % K

    # 4. Build cluster → allowed channels mapping
    cluster_channels = []
    for k in range(K):
        start = k * C
        ch_list = list(range(start, start + C))
        cluster_channels.append(ch_list)

    return clusters, cluster_channels

# ----------------------------------------------------
# THOMPSON SAMPLING DYNAMIC CHANNEL SELECTION
# ----------------------------------------------------


LAMBDA_LRU = 0.25
LAMBDA_OVER = 0.4
def dynamic_select(allowed_channels, now, observed_stats):

    candidate_channels = list(allowed_channels)

    # ---------- 1. Extract context ----------
    lru_scores = {}
    load_fracs = {}
    coll_rates = {}

    total_trans = sum(
        max(observed_stats[ch]['transmissions'], 0)
        for ch in candidate_channels
    )

    for ch in candidate_channels:
        st = observed_stats[ch]

        # LRU (larger = better)
        if st['last_used'] > 0:
            lru_scores[ch] = now - st['last_used']
        else:
            lru_scores[ch] = now

        # Load fraction (smaller = better)
        load_fracs[ch] = (
            st['transmissions'] / total_trans
            if total_trans > 0 else 0.0
        )

        # Collision rate (smaller = better)
        if st['transmissions'] > 0:
            coll_rates[ch] = st['collisions'] / st['transmissions']
        else:
            coll_rates[ch] = 0.0

    max_lru = max(lru_scores.values()) if candidate_channels else 1.0

    # ---------- 2. Thompson Sampling base ----------
    theta_raw = []
    for ch in candidate_channels:
        theta = np.random.beta(
            simulation.ts_alpha[ch],
            simulation.ts_beta[ch]
        )
        theta_raw.append(theta)

    theta_raw = np.array(theta_raw)

    # ---------- 3. Scoring system (α, β, γ) ----------
    score_components = []
    for ch in candidate_channels:

        norm_lru  = lru_scores[ch] / max_lru
        load      = load_fracs[ch]
        coll      = coll_rates[ch]

        S = alpha * norm_lru - beta * load - gamma * coll

        score_components.append(S)

    score_components = np.array(score_components)

    # ---------- 4. Combine scoring + TS ----------
    theta_final = theta_raw + lambda_val * score_components

    # ---------- 5. Pick best channel ----------
    chosen_idx = np.argmax(theta_final)
    chosen_channel = candidate_channels[chosen_idx]

    # Return feature logs
    features = [
        lru_scores[chosen_channel],
        load_fracs[chosen_channel],
        coll_rates[chosen_channel],
        score_components[chosen_idx]
    ]

    return chosen_channel, features, theta_final[chosen_idx], 1.0




def random_select(allowed_channels, now, observed_stats):
    """
    Baseline: choose a channel uniformly at random
    from the allowed set. Ignores all stats.
    """
    ch = np.random.choice(allowed_channels)

    # dummy info for compatibility
    feat = [0.0, 0.0, 0.0]
    score = 0.0
    prob = 1.0 / len(allowed_channels) if len(allowed_channels) > 0 else 0.0

    return ch, feat, score, prob
