import numpy as np
from simulation import channel_stats
# ----------------------------------------------------
# STATIC CLUSTER ASSIGNMENT BASED ON DISTANCE/RX
# ----------------------------------------------------
def assign_clusters_distance_based(d, RX, C, NUM_CHANNELS=8):
    """
    Distance/RX-based static cluster assignment.
    d: distances array of shape (N,)
    RX: received power array of shape (N,)
    C: number of channels per cluster (1, 2, or 4)
    Returns:
      clusters: array mapping each device to cluster index
      cluster_channels: list of allowed channels per cluster
    """

    N = len(d)
    K = NUM_CHANNELS // C  # number of clusters

    # 1. Sort devices by received power (strong → weak)
    sorted_idx = np.argsort(-RX)   # negative to sort descending

    # 2. Create cluster array
    clusters = np.zeros(N, dtype=int)

    # 3. Balanced interleaving:
    # Assign strongest, then weakest, then second strongest, then second weakest, etc.
    i = 0                    # pointer for strongest
    j = N - 1                # pointer for weakest
    assign_order = []

    while i <= j:
        assign_order.append(sorted_idx[i])
        if i != j:
            assign_order.append(sorted_idx[j])
        i += 1
        j -= 1

    # Now assign devices in round-robin across K clusters
    for pos, dev_idx in enumerate(assign_order):
        clusters[dev_idx] = pos % K

    # 4. Build cluster channel lists
    cluster_channels = []
    for k in range(K):
        start = k * C
        ch_list = list(range(start, start + C))
        cluster_channels.append(ch_list)

    return clusters, cluster_channels

# ----------------------------------------------------
# DYNAMIC CHANNEL SELECTION Algorithim
# ----------------------------------------------------

def dynamic_select(allowed_channels, now):
    """
    Combined dynamic channel selection:
    LRU + load-aware + collision-aware (weighted).
    """

    scores = []
    for ch in allowed_channels:
        stats = channel_stats[ch]

        # LRU term
        lru_score = now - stats['last_used']

        # Load term (penalize channels with many recent transmissions)
        load_score = -(stats['transmissions'] / (sum(channel_stats[c]['transmissions'] for c in allowed_channels) + 1))

        # Collision term (penalize channels with high collision ratio)
        if stats['transmissions'] == 0:
            collision_score = 0
        else:
            collision_score = -(stats['collisions'] / stats['transmissions'])

        # Weighted sum
        alpha = 1.0
        beta = 0.5
        gamma = 1.5

        score = alpha * lru_score + beta * load_score + gamma * collision_score
        scores.append(score)

    # Softmax to convert scores → probabilities
    lambda_val = 1.0
    exp_scores = np.exp(lambda_val * np.array(scores))
    probs = exp_scores / np.sum(exp_scores)
    # similar to RL 
    # Randomly choose channel according to probabilities
    chosen = np.random.choice(allowed_channels, p=probs)
    return chosen