import numpy as np

from waveforms.waveform import Waveform


class TwoSinesWaveform(Waveform):
    def __init__(
        self,
        c1=0.5,
        c2=0.5,
        f1=10.0,
        f2=20.0,
        max_amplitude=500.0,
    ):
        super().__init__(max_amplitude)
        self.c1 = c1
        self.c2 = c2
        self.f1 = f1
        self.f2 = f2

        self.max_amplitude = max_amplitude

    @property
    def n_params(self):
        return 4  # c1, c2, f1, f2

    @property
    def param_bounds(self):
        return {
            "c1": (0.0, 200.0),
            "c2": (0.0, 200.0),
            "f1": (0.001, 100.0),  # Hz
            "f2": (0.001, 100.0),  # Hz
        }

    def _resolve_params(self, params):
        params = {
            "c1": params.get("c1", self.normalize_param(self.c1, "c1")),
            "c2": params.get("c2", self.normalize_param(self.c2, "c2")),
            "f1": params.get("f1", self.normalize_param(self.f1, "f1")),
            "f2": params.get("f2", self.normalize_param(self.f2, "f2")),
        }

        unnormalized_params = {
            key: self.unnormalize_model_param(params[key], key) for key in params
        }
        unnormalized_params["delay"] = 0.0
        return unnormalized_params

    def _return_params(self, unnormalized_params):
        return {
            key: self.normalize_param(unnormalized_params[key], key)
            for key in unnormalized_params
            if key != "delay"
        }

    def _is_active(self, t, params):
        return True

    def _compute_value(self, pulse_t, params):
        t = pulse_t
        c1 = params["c1"]
        c2 = params["c2"]
        f1 = params["f1"]
        f2 = params["f2"]

        t_s = t / 1000.0  # convert ms to seconds for Hz-based frequencies
        return c1 * np.sin(2 * np.pi * f1 * t_s) + c2 * np.sin(2 * np.pi * f2 * t_s)
