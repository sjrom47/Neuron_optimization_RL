import numpy as np
from scipy.signal import find_peaks


def firing_rate(membrane_potential, t):
    num_peaks, _ = find_peaks(membrane_potential, prominence=40)
    duration = np.max(t) / 1000  # convert to seconds
    firing_rate = len(num_peaks) / duration  # firing rate
    return firing_rate
