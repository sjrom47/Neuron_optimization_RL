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

        active_interval = []
        active_t_points = []
        # Extract active interval indexes
        for i, t in enumerate(t_points):
            pulse_t = t - p["delay"] / 1000.0
            if self._is_active(pulse_t, p):
                active_interval.append(i)
                active_t_points.append(pulse_t)
            elif active_interval:
                t_interval = np.array(active_t_points)
                i_interval = np.array(active_interval)
                waveform[i_interval] = np.clip(
                    self._compute_value(t_interval, p),
                    -self.max_amplitude,
                    self.max_amplitude,
                )
                active_interval = []
                active_t_points = []
        if active_interval:
            t_interval = np.array(active_t_points)
            i_interval = np.array(active_interval)
            waveform[i_interval] = np.clip(
                self._compute_value(t_interval, p),
                -self.max_amplitude,
                self.max_amplitude,
            )

        return waveform, self._return_params(p)

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
