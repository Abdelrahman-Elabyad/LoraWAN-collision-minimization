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
alpha = 0.2      # Prioritizes load balancing via LRU
beta = 0.01     # Moderate penalty for overall channel load
gamma = 0.3    # Heaviest penalty for direct failure evidence (collisions)
lambda_val = 0.1 # Aggressive exploitation (greedy selection)


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

# Small prior weights: gentle, not dominant
LAMBDA_LRU  = 0.01   # how much to reward long-unvisited channels
LAMBDA_OVER = 0.01   # how much to penalize overloaded channels


def dynamic_select(allowed_channels, now, observed_stats):
    """
    Knowledge-boosted Thompson Sampling:

    - Base: Beta(1 + successes, 1 + failures) per channel.
    - LRU prior boost: channels not used for a long time get a small alpha bonus.
    - Overload prior penalty: channels with high collision + load get a small beta bonus.

    We ONLY use stale stats to slightly bias the Beta prior,
    never to deterministically eliminate channels.

    Returns:
        chosen_channel,
        [lru_score, load_frac, coll_rate],   # features (for logging/ML)
        theta,                               # sampled quality (as "score")
        prob                                # softmax prob for logging
    """

    # ---------- 1) PRECOMPUTE LRU & LOAD/COLL FEATURES ----------
    # Total transmissions on allowed channels (for load fraction)
    total_trans = 0
    for ch in allowed_channels:
        total_trans += observed_stats[ch]['transmissions']
    if total_trans <= 0:
        total_trans = 0

    # First pass: compute raw lru, load_frac, coll_rate, success_rate
    lru_dict = {}
    features = {}   # ch -> (lru_score, load_frac, coll_rate, success_rate)

    for ch in allowed_channels:
        stats = observed_stats[ch]

        trans = stats['transmissions']
        succ  = stats['successes']
        coll  = stats['collisions']
        last_used = stats['last_used']

        # LRU: time since last use; if never used, treat as "now" (max)
        if last_used > 0:
            lru_score = now - last_used
        else:
            lru_score = now  # large, means "very long ago" / "never used"

        # Load fraction among allowed channels
        if total_trans > 0:
            load_frac = trans / total_trans
        else:
            load_frac = 0.0

        # Collision & success rates
        if trans > 0:
            coll_rate    = coll / trans
            success_rate = succ / trans
        else:
            coll_rate    = 0.0
            success_rate = 0.0

        lru_dict[ch] = lru_score
        features[ch] = (lru_score, load_frac, coll_rate, success_rate)

    # Normalize LRU for prior boost
    max_lru = max(lru_dict[ch] for ch in allowed_channels) if allowed_channels else 0.0
    if max_lru <= 0:
        max_lru = 1.0

    # ---------- 2) BUILD BETA PRIORS WITH SMALL KNOWLEDGE BIASES ----------
    thetas = []
    feature_list = []
    scores = []
    candidate_channels = list(allowed_channels)  # just a local copy

    for ch in candidate_channels:
        stats = observed_stats[ch]
        lru_score, load_frac, coll_rate, success_rate = features[ch]

        trans = stats['transmissions']
        succ  = stats['successes']
        failures = max(0, trans - succ)

        # Base Beta counts from successes/failures
        alpha_0 = 1.0 + succ
        beta_0  = 1.0 + failures

        # ---- LRU prior boost (exploration) ----
        # Normalized LRU in [0, 1]; longer-unvisited => closer to 1
        norm_lru = lru_score / max_lru
        eps_lru = LAMBDA_LRU * norm_lru

        # ---- Overload prior penalty (stale but useful) ----
        overload = 0.7 * coll_rate + 0.3 * load_frac
        eps_over = LAMBDA_OVER * overload

        # Final Beta parameters
        alpha = alpha_0 + eps_lru       # reward for being idle for long
        beta  = beta_0  + eps_over      # penalty for looking overloaded

        # Safety clamp
        alpha = max(alpha, 1e-6)
        beta  = max(beta, 1e-6)

        # ---- Thompson Sampling draw ----
        theta = np.random.beta(alpha, beta)

        thetas.append(theta)
        scores.append(theta)
        # Expose the main features for optional logging/ML
        feature_list.append([lru_score, load_frac, coll_rate])

    thetas = np.array(thetas)

    # ---------- 3) SOFTMAX OVER THETAS FOR A PROBABILITY VECTOR ----------
    max_theta = np.max(thetas)
    exp_vals = np.exp(thetas - max_theta)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        probs = np.ones_like(exp_vals) / len(exp_vals)
    else:
        probs = exp_vals / sum_exp

    # Sample channel index according to these probabilities
    chosen_idx = np.random.choice(len(candidate_channels), p=probs)
    chosen_channel = candidate_channels[chosen_idx]

    chosen_features = feature_list[chosen_idx]
    chosen_score = scores[chosen_idx]
    chosen_prob = probs[chosen_idx]

    return chosen_channel, chosen_features, chosen_score, chosen_prob

