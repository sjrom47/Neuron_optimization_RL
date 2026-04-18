import numpy as np
from scipy.signal import find_peaks


def firing_rate(membrane_potential, t):
    peaks, _ = find_peaks(membrane_potential, prominence=40)
    if len(t) < 2:
        return 0.0

    duration = float(np.max(t) - np.min(t))
    if duration <= 0.0:
        return 0.0

    # Auto-handle either seconds or milliseconds time vectors.
    duration_s = duration / 1000.0 if duration > 10.0 else duration
    return len(peaks) / max(duration_s, 1e-9)
