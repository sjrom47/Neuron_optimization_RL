import numpy as np

from config import STIMULATION_DURATION
from waveforms.waveform import Waveform


class ChargeBalancedWaveform(Waveform):
    def __init__(
        self,
        t1=5.0,
        t2=5.0,
        amplitude=100.0,
        n_cycles=5.0,
        max_amplitude=500.0,
    ):
        super().__init__(max_amplitude)
        self.t1 = t1
        self.t2 = t2
        self.amplitude = min(amplitude, max_amplitude)
        self.n_cycles = n_cycles
        self.delay = 0.0

    @property
    def n_params(self):
        return 4  # t1, t2, amplitude, n_cycles

    @property
    def param_bounds(self):
        return {
            "t1": (0.05, 0.95),
            "t2": (0.05, 0.95),
            "amplitude": (0.0, self.max_amplitude),
            "n_cycles": (1.0, STIMULATION_DURATION),
            "delay": (
                0.0,
                STIMULATION_DURATION,
            ),  # cosmetic — always 0, required by base class
        }

    def _resolve_params(self, params):
        params = {
            "t1": params.get("t1", self.normalize_param(self.t1, "t1")),
            "t2": params.get("t2", self.normalize_param(self.t2, "t2")),
            "amplitude": params.get(
                "amplitude", self.normalize_param(self.amplitude, "amplitude")
            ),
            "n_cycles": params.get(
                "n_cycles", self.normalize_param(self.n_cycles, "n_cycles")
            ),
            "delay": self.normalize_param(self.delay, "delay"),
        }
        unnormalized_params = {
            key: self.unnormalize_model_param(params[key], key) for key in params
        }
        return unnormalized_params

    def _is_active(self, t, params):
        period = STIMULATION_DURATION / params["n_cycles"]
        return (t >= 0) & (
            t % period
            < (period * params["t1"]) + period * (1 - params["t1"]) * params["t2"]
        )

    def _compute_value(self, pulse_t, params):
        period = STIMULATION_DURATION / params["n_cycles"]
        t_in_cycle = pulse_t % period
        t1, t2, a1 = params["t1"], params["t2"], params["amplitude"]

        # Charge balance: a1*t1 = a2*(1-t1)*t2  →  a2 = a1*t1/((1-t1)*t2)
        # If a2 would exceed max_amplitude, scale both phases down to stay balanced.
        denom = (1 - t1) * t2
        if denom < 1e-9:
            return np.full_like(pulse_t, 0.0)
        a2_required = a1 * t1 / denom
        if a2_required > self.max_amplitude:
            a2 = self.max_amplitude
            a1 = self.max_amplitude * (1 - t1) * t2 / t1
        else:
            a2 = a2_required

        return np.where(
            t_in_cycle < t1 * period,
            a1,
            np.where(
                t_in_cycle < t1 * period + (1 - t1) * t2 * period,
                -a2,
                0.0,
            ),
        )
