from abc import ABC, abstractmethod

import numpy as np


class Waveform(ABC):
    def __init__(self, max_amplitude=2000.0):
        self.max_amplitude = max_amplitude

    def generate_waveform(self, duration, sampling_rate, params=None):
        if params is None:
            params = {"delay": 0.0}  # default delay of 0 ms if not provided

        # Extract parameters and move to actual parameter bounds
        p = self._resolve_params(params)

        # Time points
        t_points = self.get_t_points(duration, sampling_rate)
        waveform = np.zeros_like(t_points)

        intervals = []
        active_interval = []
        # Extract active interval indexes
        for t in t_points:
            pulse_t = t - p["delay"] / 1000.0
            if self._is_active(pulse_t, p):
                active_interval.append(t)
            else:
                intervals.append(np.array(active_interval))
                active_interval = []
        if active_interval:
            intervals.append(np.array(active_interval))

        for t_interval in enumerate(intervals):
            waveform[t_interval] = np.clip(
                self._compute_value(t_interval, p),
                -self.max_amplitude,
                self.max_amplitude,
            )

        return waveform, p

    @abstractmethod
    def _resolve_params(self, params):
        pass

    @abstractmethod
    def _is_active(self, pulse_t, params):
        pass

    @abstractmethod
    def _compute_value(self, pulse_t, params):
        pass

    @property
    @abstractmethod
    def n_params(self):
        pass

    @property
    @abstractmethod
    def param_bounds(self):
        pass

    def get_t_points(self, duration, sampling_rate):
        return np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    def unnormalize_model_param(self, model_param, param_name):
        bounds = self.param_bounds[param_name]
        if isinstance(bounds, list):
            return [
                self.unnormalize_model_param(mp, f"{param_name}_{i}")
                for i, mp in enumerate(model_param)
            ]
        else:
            min_val, max_val = bounds
            # Right now assumimg model output is in [0, 1], but this can be changed
            return model_param * (max_val - min_val) + min_val
