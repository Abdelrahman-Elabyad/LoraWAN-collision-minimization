import numpy as np
import json
import simulation 

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
    """
    Improved Thompson Sampling with probabilistic selection.
    """
    candidate_channels = list(allowed_channels)
    n_channels = len(candidate_channels)
    
    if n_channels == 1:
        ch = candidate_channels[0]
        return ch, [0, 0, 0, 0], 0.5, 1.0

    # ---------- 1. Extract context ----------
    lru_scores = {}
    load_fracs = {}
    success_rates = {}  # Changed from collision rates

    total_trans = sum(
        max(observed_stats[ch]['transmissions'], 1)
        for ch in candidate_channels
    )

    for ch in candidate_channels:
        st = observed_stats[ch]

        # LRU
        if st['last_used'] > 0:
            lru_scores[ch] = now - st['last_used']
        else:
            lru_scores[ch] = now

        # Load fraction
        load_fracs[ch] = st['transmissions'] / total_trans

        # SUCCESS RATE (changed from collision rate)
        if st['transmissions'] > 0:
            success_rates[ch] = st['successes'] / st['transmissions']
        else:
            success_rates[ch] = 0.5  # Unknown = neutral

    max_lru = max(lru_scores.values()) if lru_scores else 1.0
    max_lru = max(max_lru, 1.0)

    # ---------- 2. Thompson Sampling ----------
    theta_samples = []
    for ch in candidate_channels:
        alpha = observed_stats[ch]['ts_alpha']
        beta = observed_stats[ch]['ts_beta']
        theta = np.random.beta(alpha, beta)
        theta_samples.append(theta)

    theta_samples = np.array(theta_samples)

    # ---------- 3. Contextual scoring ----------
    alpha_weight = 0.15   # LRU weight
    beta_weight = 0.20    # SUCCESS rate weight (positive signal)
    gamma_weight = 0.10   # Load penalty weight
    lambda_val = 0.4

    score_components = []
    for ch in candidate_channels:
        norm_lru = lru_scores[ch] / max_lru
        load = load_fracs[ch]
        success = success_rates[ch]

        # Contextual score: reward success, penalize load
        S = alpha_weight * norm_lru + beta_weight * success - gamma_weight * load
        score_components.append(S)

    score_components = np.array(score_components)
    
    # Normalize to [-1, 1]
    if score_components.max() - score_components.min() > 0:
        score_components = (score_components - score_components.min()) / \
                          (score_components.max() - score_components.min())
        score_components = 2 * score_components - 1
    else:
        score_components = np.zeros_like(score_components)

    # ---------- 4. Combine ----------
    combined_scores = theta_samples + lambda_val * score_components

    # ---------- 5. Probabilistic selection ----------
    temperature = 0.4  # Increased from 0.3 for more exploration
    exp_scores = np.exp(combined_scores / temperature)
    probabilities = exp_scores / exp_scores.sum()

    chosen_idx = np.random.choice(len(candidate_channels), p=probabilities)
    chosen_channel = candidate_channels[chosen_idx]

    features = [
        lru_scores[chosen_channel],
        load_fracs[chosen_channel],
        success_rates[chosen_channel],
        score_components[chosen_idx]
    ]

    return chosen_channel, features, combined_scores[chosen_idx], probabilities[chosen_idx]

def dynamic_select(allowed_channels, now, observed_stats, epsilon=None):
    """
    Adaptive epsilon-greedy with load-based exploration.
    At high loads (many transmissions), increase exploration to combat stale data.
    """
    candidate_channels = list(allowed_channels)
    
    if len(candidate_channels) == 1:
        ch = candidate_channels[0]
        return ch, [0, 0, 0, 0], 0.5, 1.0
    
    # ADAPTIVE EPSILON: Increase with network load
    if epsilon is None:
        # Count total transmissions across all candidate channels
        total_trans = sum(max(observed_stats[ch]['transmissions'], 0) 
                         for ch in candidate_channels)
        
        # Base epsilon + load-based increase
        base_epsilon = 0.25
        load_bonus = min(total_trans / 5000.0, 0.25)  # Up to +0.25
        epsilon = base_epsilon + load_bonus
        
        # Cap at 0.5 (50% exploration max)
        epsilon = min(epsilon, 0.5)
    
    # Epsilon-greedy decision
    if np.random.random() < epsilon:
        # EXPLORE: Random selection
        chosen_channel = np.random.choice(candidate_channels)
        return chosen_channel, [0, 0, 0, 0], 0.5, 1.0 / len(candidate_channels)
    
    # EXPLOIT: Use Thompson Sampling
    return dynamic_select(allowed_channels, now, observed_stats)



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
