from abc import ABC, abstractmethod

import numpy as np


class Waveform(ABC):
    def generate_waveform(self, duration, sampling_rate):
        return self._generate_waveform_points(duration, sampling_rate, params={})

    @abstractmethod
    def _generate_waveform_points(self, duration, sampling_rate, params):
        pass

    @abstractmethod
    def default_stimulation(self):
        pass

    @property
    @abstractmethod
    def n_params(self):
        pass

    def get_t_points(self, duration, sampling_rate):
        return np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
