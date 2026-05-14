import time
from abc import ABC, abstractmethod

import numpy as np


class Waveform(ABC):
    def __init__(self, max_amplitude=500.0):
        self.max_amplitude = max_amplitude

    def generate_waveform(self, duration, sampling_rate, params=None):
        duration_s = duration / 1000.0  # convert ms to seconds
        t_points = self.get_t_points(duration_s, sampling_rate) * 1000.0  # convert to ms
        if params is None:
            params = {"delay": 0.0}

        p = self._resolve_params(params)

        pulse_t = t_points - p["delay"]

        # Vectorized active mask

        active_mask = self._is_active(pulse_t, p)  # returns boolean array

        waveform = np.zeros_like(t_points)
        if np.any(active_mask):
            waveform[active_mask] = np.clip(
                self._compute_value(pulse_t[active_mask], p),
                -self.max_amplitude,
                self.max_amplitude,
            )

        returned_params = self._return_params(p)

        return waveform, returned_params

    @abstractmethod
    def _resolve_params(self, params):
        pass

    def _return_params(self, unnormalized_params):
        return {
            key: self.normalize_param(unnormalized_params[key], key)
            for key in unnormalized_params
        }

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
            # Right now assumimg model output is in [-1, 1], but this can be changed
            return (model_param + 1) * (max_val - min_val) / 2 + min_val

    def normalize_param(self, param_value, param_name):
        bounds = self.param_bounds[param_name]
        if isinstance(bounds, list):
            return [
                self.normalize_param(pv, f"{param_name}_{i}")
                for i, pv in enumerate(param_value)
            ]
        else:
            min_val, max_val = bounds
            return (param_value - min_val) * 2 / (max_val - min_val) - 1
