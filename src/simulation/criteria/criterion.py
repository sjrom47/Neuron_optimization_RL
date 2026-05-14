from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import find_peaks


class Criterion(ABC):
    @abstractmethod
    def evaluate(self, response):
        pass

    @property
    @abstractmethod
    def requires_multiple_responses(self):
        pass

    @staticmethod
    def calculate_n_spikes(signal):
        signal = np.asarray(signal)
        if not signal.flags.writeable:
            signal = signal.copy()
        peaks, _ = find_peaks(signal, prominence=40)
        return len(peaks)
