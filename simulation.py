import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import heapq

# ----------------------------------------------------
RADIUS = 1.0                    # km
NUM_CHANNELS = 8
PACKET_DURATION = 0.050         # 50 ms
MEAN_INTERARRIVAL = 600         # seconds per device
TOTAL_PACKETS = 10000
success_distances = []
failure_distances = []
# -------- Information model configuration --------
INFO_MODE = "HYBRID"      # "GLOBAL_DELAY" or "PER_DEVICE_DELAY" or "HYBRID"
GLOBAL_UPDATE_INTERVAL = 2.0   # seconds
DEVICE_UPDATE_DELAY = 1    # seconds after each device's last uplink
# -------------------------------------------------

# These are module-level STATE VARIABLES
stale_stats = None
last_global_update = 0.0
device_stats = None
last_device_update = None

# Global statistics for dynamic channel selection
channel_stats = {
    ch: {
        "last_used": 0.0,
        "transmissions": 0,
        "collisions": 0,
        "successes": 0,

        # Thompson Sampling counters
        "ts_alpha": 1.0,
        "ts_beta": 1.0,

        # NEW: long-term learning fields
        "ema_succ": 0.5,   # exponential moving average of success
        "ema_coll": 0.0,   # exponential moving average of collision indicator
        "trend": 0.0,      # trend of success probability
        "last_p_ts": 0.5,  # last TS success estimate
    }
    for ch in range(NUM_CHANNELS)
}



def reset_channel_stats():
    global channel_stats
    for ch in range(NUM_CHANNELS):
        channel_stats[ch] = {
            "last_used": 0.0,
            "transmissions": 0,
            "collisions": 0,
            "successes": 0,
            "ts_alpha": 1.0,
            "ts_beta": 1.0,
            "ema_succ": 0.5,
            "ema_coll": 0.0,
            "trend": 0.0,
            "last_p_ts": 0.5,
        }


# ----------------------------------------------------
# DEVICE GENERATION
# ----------------------------------------------------
def generate_devices(N):
    """
    Generate N devices uniformly in a disk of radius RADIUS.
    Returns:
        d  = distances from gateway (km)
        RX = received power (dBm)
        pos = Nx2 array of coordinates [[x1,y1], ...]
    """

    # Uniform distribution in a circle
    U = np.random.rand(N)
    r = np.sqrt(U) * RADIUS
    theta = 2 * np.pi * np.random.rand(N)

    # Convert polar → Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    positions = np.column_stack([x, y])

    # Distance in km
    d = r

    # Convert to meters for path loss
    d_m = d * 1000

    # Path loss model
    PL = 40 + 27 * np.log10(d_m + 1e-9)   # avoid log(0)

    # Transmit power
    TP = 5   # dBm
    RX = TP - PL

    return d, RX, positions

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
def check_collision_batch(packets_on_channel):
    """
    Batch collision evaluation - processes all overlapping packets together.
    This is the new collision detection that replaces the old incremental approach.
    
    Args:
        packets_on_channel: List of packet dicts with 'start', 'end', 'rx', 'success' keys
    
    Returns:
        None (modifies packets in-place by setting 'success' field)
    """
    if not packets_on_channel:
        return
    
    # Sort by start time
    packets_on_channel.sort(key=lambda p: p['start'])
    
    # Find overlapping groups
    i = 0
    while i < len(packets_on_channel):
        group = [packets_on_channel[i]]
        end_time = packets_on_channel[i]['end']
        
        # Collect all overlapping packets
        j = i + 1
        while j < len(packets_on_channel) and packets_on_channel[j]['start'] < end_time:
            group.append(packets_on_channel[j])
            end_time = max(end_time, packets_on_channel[j]['end'])
            j += 1
        
        # Evaluate this group
        if len(group) == 1:
            # No collision - packet succeeds
            group[0]['success'] = True
        else:
            # Collision scenario - check for capture effect
            powers = [p['rx'] for p in group]
            max_rx = max(powers)
            
            # Check if strongest dominates by 3dB
            capture_ok = all(max_rx - rx >= 3.0 for rx in powers if rx != max_rx)
            
            if capture_ok:
                # Strongest wins
                winner_idx = powers.index(max_rx)
                group[winner_idx]['success'] = True
            # else: all fail (default is False)
        
        i = j


def fast_random_simulation(N, distances, RX, clusters, cluster_channels, arrivals, dev_sequence):
    """
    Pure random channel selection with batch collision detection.
    NOW uses realistic batch collision evaluation like the competitor.
    Tracks distances for CDF plots.
    """
    NUM_CHANNELS = len(cluster_channels) * len(cluster_channels[0])
    
    # Generate all packets first
    packets_by_channel = {ch: [] for ch in range(NUM_CHANNELS)}
    
    for t, dev in zip(arrivals, dev_sequence):
        cluster_id = clusters[dev]
        allowed = cluster_channels[cluster_id]
        
        # Pure random channel selection
        ch = np.random.choice(allowed)
        
        # Create packet
        packet = {
            'device_id': dev,
            'start': t,
            'end': t + PACKET_DURATION,
            'rx': RX[dev],
            'distance': distances[dev],
            'success': False  # Will be updated by collision detection
        }
        
        packets_by_channel[ch].append(packet)
    
    # Evaluate collisions using batch processing
    for ch in range(NUM_CHANNELS):
        check_collision_batch(packets_by_channel[ch])
    
    # Collect results
    all_packets = []
    for ch in range(NUM_CHANNELS):
        all_packets.extend(packets_by_channel[ch])
    
    success = sum(1 for p in all_packets if p['success'])
    success_distances = [p['distance'] for p in all_packets if p['success']]
    failure_distances = [p['distance'] for p in all_packets if not p['success']]
    
    return success / len(all_packets), success_distances, failure_distances


def run_simulation(N, distances, RX, clusters, cluster_channels, select_channel_fn, arrivals, dev_sequence):
    """
    Main simulation function with batch collision detection.
    Works with dynamic channel selection functions.
    """
    if select_channel_fn.__name__ == "random_select":
        return fast_random_simulation(N, distances, RX, clusters, cluster_channels, arrivals, dev_sequence)

    global stale_stats, last_global_update, device_stats, last_device_update
    
    # Reset stats
    reset_channel_stats()

    # Initialize STALE VIEWS
    stale_stats = copy.deepcopy(channel_stats)
    last_global_update = 0.0

    device_stats = [copy.deepcopy(channel_stats) for _ in range(N)]
    last_device_update = [0.0 for _ in range(N)]

    NUM_CHANNELS = len(cluster_channels) * len(cluster_channels[0])
    packets_by_channel = {ch: [] for ch in range(NUM_CHANNELS)}
    event_queue = []

    # Insert periodic global updates (Class B downlinks)
    if INFO_MODE in ["GLOBAL_DELAY", "HYBRID"]:
        next_global = GLOBAL_UPDATE_INTERVAL
        sim_end = arrivals[-1] + 10.0
        while next_global < sim_end:
            heapq.heappush(event_queue, (next_global, "global_update", None))
            next_global += GLOBAL_UPDATE_INTERVAL

    # Insert packet arrivals
    for t, dev in zip(arrivals, dev_sequence):
        heapq.heappush(event_queue, (t, "arrival", dev))

    # -------------- MAIN EVENT LOOP (for channel selection) --------------
    while event_queue:
        t, event_type, dev = heapq.heappop(event_queue)

        # -------- DEVICE UPDATE EVENT --------
        if event_type == "dev_update":
            device_stats[dev] = copy.deepcopy(channel_stats)
            last_device_update[dev] = t
            continue

        # -------- GLOBAL UPDATE EVENT --------
        if event_type == "global_update":
            stale_stats = copy.deepcopy(channel_stats)
            last_global_update = t
            continue

        # -------- PACKET ARRIVAL EVENT --------
        if event_type == "arrival":
            d_dev = distances[dev]
            cluster_id = clusters[dev]
            allowed = cluster_channels[cluster_id]

            # Determine which stale view to use
            if INFO_MODE == "GLOBAL_DELAY":
                observed_stats = stale_stats
            elif INFO_MODE == "PER_DEVICE_DELAY":
                observed_stats = device_stats[dev]
            elif INFO_MODE == "HYBRID":
                if last_device_update[dev] >= last_global_update:
                    observed_stats = device_stats[dev]
                else:
                    observed_stats = stale_stats
            else:
                raise ValueError("Invalid INFO_MODE")

            # Channel selection (uses STALE data)
            channel, feat, score, prob = select_channel_fn(allowed, t, observed_stats, N)

            # Create packet
            packet = {
                'device_id': dev,
                'start': t,
                'end': t + PACKET_DURATION,
                'rx': RX[dev],
                'distance': d_dev,
                'success': False
            }
            
            packets_by_channel[channel].append(packet)

            # Update ground truth stats (for future selections)
            channel_stats[channel]['transmissions'] += 1
            channel_stats[channel]['last_used'] = t

            # Schedule device update (ACK downlink)
            update_time = t + DEVICE_UPDATE_DELAY
            heapq.heappush(event_queue, (update_time, "dev_update", dev))

    # -------------- BATCH COLLISION EVALUATION --------------
    for ch in range(NUM_CHANNELS):
        check_collision_batch(packets_by_channel[ch])
    
    # Collect results and update statistics
    all_packets = []
    for ch in range(NUM_CHANNELS):
        all_packets.extend(packets_by_channel[ch])
        
        # Update final statistics based on outcomes
        for packet in packets_by_channel[ch]:
            if packet['success']:
                channel_stats[ch]['successes'] += 1
                channel_stats[ch]['ts_alpha'] += 1
            else:
                channel_stats[ch]['collisions'] += 1
                channel_stats[ch]['ts_beta'] += 1
            
            # Update long-term EMA metrics
            inst_succ = 1.0 if packet['success'] else 0.0
            inst_coll = 0.0 if packet['success'] else 1.0
            
            BETA_SUCC = 0.02
            BETA_COLL = 0.02
            BETA_TREND = 0.05
            
            channel_stats[ch]['ema_succ'] = \
                (1.0 - BETA_SUCC) * channel_stats[ch]['ema_succ'] + BETA_SUCC * inst_succ
            
            channel_stats[ch]['ema_coll'] = \
                (1.0 - BETA_COLL) * channel_stats[ch]['ema_coll'] + BETA_COLL * inst_coll
            
            p_ts_old = channel_stats[ch]['last_p_ts']
            p_ts_new = channel_stats[ch]['ts_alpha'] / \
                      (channel_stats[ch]['ts_alpha'] + channel_stats[ch]['ts_beta'])
            
            delta_p = p_ts_new - p_ts_old
            channel_stats[ch]['trend'] = \
                (1.0 - BETA_TREND) * channel_stats[ch]['trend'] + BETA_TREND * delta_p
            
            channel_stats[ch]['last_p_ts'] = p_ts_new
    
    success = sum(1 for p in all_packets if p['success'])
    success_distances = [p['distance'] for p in all_packets if p['success']]
    failure_distances = [p['distance'] for p in all_packets if not p['success']]
    
    return success / len(all_packets), success_distances, failure_distances

# ----------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------
def compute_cdf(values):
    if values is None or len(values) == 0:
        return None, None

    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

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


