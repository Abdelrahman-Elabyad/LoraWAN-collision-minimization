import numpy as np
import json
from simulation import channel_stats, USE_ML


# Recommended static parameters for maximum success (without ML)
# Old default static parameters
# alpha = 1.0
# beta = 0.5
# gamma = 1.5
# lambda_val = 1.0

# NEW Recommended parameters
alpha = 1.5      # Prioritizes load balancing via LRU
beta = 0.75      # Moderate penalty for overall channel load
gamma = 2.0      # Heaviest penalty for direct failure evidence (collisions)
lambda_val = 2.0 # Aggressive exploitation (greedy selection)

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
# DYNAMIC CHANNEL SELECTION Algorithim
# ----------------------------------------------------

def dynamic_select(allowed_channels, now, observed_stats):
    scores = []
    feature_list = []

    for ch in allowed_channels:
        stats = observed_stats[ch]   

        lru_score = now - stats['last_used']

        load_score = -(stats['transmissions'] /
                        (sum(observed_stats[c]['transmissions'] for c in allowed_channels) + 1))

        if stats['transmissions'] == 0:
            collision_score = 0
        else:
            collision_score = -(stats['collisions'] / stats['transmissions'])

        feature_list.append([lru_score, load_score, collision_score])

        score = alpha*lru_score + beta*load_score + gamma*collision_score
        scores.append(score)
    scores = np.array(scores)

    # --- Numerically stable softmax ---
    max_score = np.max(scores)                    # subtract the max for stability
    exp_scores = np.exp(lambda_val * (scores - max_score))

    sum_exp = np.sum(exp_scores)
    if sum_exp == 0:                              # fallback (rare safeguard)
        probs = np.ones_like(exp_scores) / len(exp_scores)
    else:
        probs = exp_scores / sum_exp
    # --- End softmax fix ---

    chosen_idx = np.random.choice(len(allowed_channels), p=probs)
    chosen_channel = allowed_channels[chosen_idx]

    chosen_features = feature_list[chosen_idx]
    chosen_score = scores[chosen_idx]
    chosen_prob = probs[chosen_idx]

    return chosen_channel, chosen_features, chosen_score, chosen_prob

