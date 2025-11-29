import numpy as np
import math
import matplotlib.pyplot as plt
import os
import csv

# ----------------------------------------------------
RADIUS = 1.0                    # km
NUM_CHANNELS = 8
PACKET_DURATION = 0.050         # 50 ms
MEAN_INTERARRIVAL = 600         # seconds per device
TOTAL_PACKETS = 10000

success_distances = []
failure_distances = []
DATASET_FILE = "channel_training_data.csv"

# Global statistics for dynamic channel selection
channel_stats = {
    ch: {
        'last_used': 0.0,
        'transmissions': 0,
        'collisions': 0,
        'successes': 0
    }
    for ch in range(NUM_CHANNELS)
}

def reset_channel_stats():
    for ch in range(NUM_CHANNELS):
        channel_stats[ch] = {
            'last_used': 0.0,
            'transmissions': 0,
            'collisions': 0,
            'successes': 0
        }

def initialize_dataset():
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", 
                "channel",
                "lru_score", 
                "load_score", 
                "collision_score",
                "score",
                "probability",
                "success",
                "distance",
                "rx",
                "alpha", "beta", "gamma", "lambda"
            ])
# ----------------------------------------------------
# DEVICE GENERATION
# ----------------------------------------------------
def generate_devices(N):
    """Generate N devices uniformly in a disk, return distances + RX."""
    U = np.random.rand(N)
    r = np.sqrt(U) * RADIUS
    theta = 2 * np.pi * np.random.rand(N)

    # Distance in km
    d = r

    # Convert to meters for PL formula
    d_m = d * 1000

    # Path loss
    PL = 40 + 27 * np.log10(d_m)

    # Transmit power
    TP = 5  # dBm
    RX = TP - PL

    return d, RX
# ----------------------------------------------------
# ARRIVAL PROCESS (AGGREGATE POISSON)
# ----------------------------------------------------
def generate_arrivals(N):
    """Return arrival timestamps for TOTAL_PACKETS arrivals."""
    rate = N / MEAN_INTERARRIVAL
    interarrival_mean = 1 / rate
    inter = np.random.exponential(scale=interarrival_mean, size=TOTAL_PACKETS)
    
    # Cumulative sum to get arrival times
    times = np.cumsum(inter) #[0.2, 0.05, 0.13, 0.01, ...] into [0.2, 0.25, 0.38, 0.39, ...]
    return times


# ----------------------------------------------------
# COLLISION & CAPTURE CHECK
# ----------------------------------------------------

def check_collision(channel_packets, timestamp, end_time, RX):
    """Apply collision/capture rule for a single new packet.
       Returns True if the NEW packet succeeds, False otherwise.
    """

    # Find overlapping packets
    # A packet overlaps if it starts before the new packet ends
    # and ends after the new packet starts.
    #this list contains all the pakcets that havent finished yet
    overlaps = [p for p in channel_packets
                if not (p['end'] <= timestamp or p['start'] >= end_time)]

    new_packet = {'start': timestamp, 'end': end_time, 'RX': RX}

    if len(overlaps) == 0:
        # No collision → new packet succeeds
        channel_packets.append(new_packet)
        return True

    # There is at least one overlapping packet → collision scenario
    all_packets = overlaps + [new_packet]
    powers = [p['RX'] for p in all_packets]
    strongest = max(powers)
    idx_strongest = powers.index(strongest)

    # Check if strongest dominates all others by >= 3 dB
    dominates = True
    for pwr in powers:
        if strongest - pwr < 3 and pwr != strongest:
            dominates = False
            break

    # Add the new packet to the ongoing list (it occupies the air)
    channel_packets.append(new_packet)

    if not dominates:
        # No packet clearly wins → treat as collision, new packet fails
        return False

    # There is exactly one dominant strongest packet.
    # NEW packet succeeds only if it is that strongest one.
    index_new = len(all_packets) - 1  # new packet is last in all_packets
    if idx_strongest == index_new:
        return True   # new packet is strongest and dominates
    else:
        return False  # some older packet was strongest; new one loses

# ----------------------------------------------------
# MAIN SIMULATION
# ----------------------------------------------------

def run_simulation(N, clusters, cluster_channels, select_channel_fn):
    global success_distances, failure_distances
    success_distances = []
    failure_distances = []
    """
    N: number of devices
    clusters: array mapping device → cluster index
    cluster_channels: list of channels for each cluster
    select_channel_fn: function(allowed_channels, now) → channel_id
    """

    # Prepare device parameters
    distances, RX = generate_devices(N)
    arrivals = generate_arrivals(N)

    # Track ongoing packets per channel
    ongoing = [[] for _ in range(NUM_CHANNELS)]

    success = 0

    for t in arrivals:

        # 1. Pick a random device
        dev = np.random.randint(0, N)
        d_dev = distances[dev]

        cluster_id = clusters[dev]
        allowed = cluster_channels[cluster_id]

        # 2. Choose channel dynamically
        channel = select_channel_fn(allowed, t)

        # 3. Remove expired (finished) packets from this channel
        ongoing[channel] = [p for p in ongoing[channel] if p['end'] > t]

        # 4. Compute end time
        end_t = t + PACKET_DURATION

        # 5. Update stats: transmission attempted
        channel_stats[channel]['transmissions'] += 1
        channel_stats[channel]['last_used'] = t

        # 6. Check collision/capture
        ok = check_collision(ongoing[channel], t, end_t, RX[dev])

        if ok:
            success += 1
            channel_stats[channel]['successes'] += 1
            success_distances.append(d_dev)
        else:
            channel_stats[channel]['collisions'] += 1
            failure_distances.append(d_dev)

    # SUCCESS PROBABILITY
    return success / TOTAL_PACKETS

# ----------------------------------------------------
# EXPERIMENTS & PLOTTING FUNCTIONS
# ----------------------------------------------------

def experiment_vary_C(N, clustering_fn, selection_fn):
    results = {}

    for C in [1, 2, 4]:
        print(f"Running experiment with C = {C}")

        # Generate devices for this experiment
        d, RX = generate_devices(N)

        # Static clustering algorithm
        clusters, cluster_channels = clustering_fn(d, RX, C)

        # Reset stats
        reset_channel_stats()

        # Run simulation
        prob = run_simulation(N, clusters, cluster_channels, selection_fn)
        results[C] = prob

        print(f"C = {C}, success = {prob}")

    return results

def plot_success_vs_C(results, title="Success vs Cluster Size"):
    Cs = sorted(results.keys())
    probs = [results[C] for C in Cs]

    plt.figure(figsize=(7,5))
    plt.plot(Cs, probs, marker='o')
    plt.xlabel("Cluster size C")
    plt.ylabel("Success probability")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_success_cdf():
    xs = np.sort(success_distances)
    ys = np.arange(1, len(xs)+1) / len(xs)

    plt.figure(figsize=(7,5))
    plt.plot(xs, ys)
    plt.xlabel("Distance (km)")
    plt.ylabel("CDF of Successful Packets")
    plt.title("Success Distance CDF")
    plt.grid(True)
    plt.show()

def plot_success_by_cluster_size(results):
    plt.figure(figsize=(10,6))

    for N in results.keys():
        Cs = sorted(results[N].keys())
        probs = [results[N][C] for C in Cs]
        plt.plot(Cs, probs, marker='o', label=f"N = {N}")

    plt.xlabel("Cluster Size C")
    plt.ylabel("Success Probability")
    plt.title("Success Probability vs Cluster Size (C) for Different N")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_success_by_device_count(results):
    plt.figure(figsize=(10,6))

    # set of C values:
    cluster_sizes = sorted(next(iter(results.values())).keys())

    for C in cluster_sizes:
        Ns = sorted(results.keys())
        probs = [results[N][C] for N in Ns]
        plt.plot(Ns, probs, marker='o', label=f"C = {C}")

    plt.xlabel("Number of Devices N")
    plt.ylabel("Success Probability")
    plt.title("Success Probability vs Device Count (N) for Different C")
    plt.grid(True)
    plt.legend()
    plt.show()


