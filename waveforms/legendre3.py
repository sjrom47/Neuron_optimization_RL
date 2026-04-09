import numpy as np
from numpy.polynomial.legendre import legval

from waveforms.waveform import Waveform


class Legendre3Waveform(Waveform):
    def __init__(
        self,
        c0=0.5,
        c1=0.5,
        c2=0.5,
        period=1.0,
        duty_cycle=0.5,
        delay=0.0,
        max_amplitude=2000.0,
    ):
        super().__init__(max_amplitude)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.period = period
        self.duty_cycle = duty_cycle
        self.delay = delay
        self.max_amplitude = max_amplitude

    @property
    def n_params(self):
        return 6  # c0, c1, c2, period, duty_cycle, delay

    @property
    def param_bounds(self):
        return {
            "c0": (-self.max_amplitude, self.max_amplitude),
            "c1": (-self.max_amplitude, self.max_amplitude),
            "c2": (-self.max_amplitude, self.max_amplitude),
            "period": (0.001, 100.0),  # seconds
            "duty_cycle": (0.0, 1.0),  # fraction of period
            "delay": (0.0, 100.0),  # milliseconds
        }

    def _resolve_params(self, params):
        c0 = params.get("c0", self.c0)
        c1 = params.get("c1", self.c1)
        c2 = params.get("c2", self.c2)
        period = params.get("period", self.period)
        duty_cycle = params.get("duty_cycle", self.duty_cycle)
        delay = params.get("delay", self.delay)

        return {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "period": period,
            "duty_cycle": duty_cycle,
            "delay": delay,
        }

    def _is_active(self, t, params):
        period = params["period"]
        duty_cycle = params["duty_cycle"]
        return (t >= 0) & ((t % period) < duty_cycle * period)

    def _compute_value(self, pulse_t, params):
        phase = (pulse_t - min(pulse_t)) / (
            max(pulse_t) - min(pulse_t)
        )  # Shift to start at 0 and end at 1 for the interval of interest
        phase_legendre_domain = 2 * phase - 1  # Map to [-1, 1] for Legendre polynomials
        legendre_values = legval(
            phase_legendre_domain, [params["c0"], params["c1"], params["c2"]]
        )
        legendre_values = legendre_values
        return legendre_values
