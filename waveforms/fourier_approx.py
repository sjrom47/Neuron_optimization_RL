import numpy as np

from waveforms.waveform import Waveform


class FourierWaveform(Waveform):
    def __init__(
        self,
        c0=0.5,
        c1=0.5,
        c2=0.5,
        f1=10.0,
        f2=20.0,
        duty_cycle=0.5,
        period=1.0,
        phase_1=0.0,
        phase_2=0.0,
        delay=0.0,
        max_amplitude=2000.0,
    ):
        super().__init__(max_amplitude)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.f1 = f1
        self.f2 = f2
        self.duty_cycle = duty_cycle
        self.period = period
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.delay = delay
        self.max_amplitude = max_amplitude

    @property
    def n_params(self):
        return 10  # c0, c1, c2, f1, f2, duty_cycle, period, phase_1, phase_2, delay

    @property
    def param_bounds(self):
        return {
            "c0": (-self.max_amplitude, self.max_amplitude),
            "c1": (-self.max_amplitude, self.max_amplitude),
            "c2": (-self.max_amplitude, self.max_amplitude),
            "f1": (0.001, 100.0),  # Hz
            "f2": (0.001, 100.0),  # Hz
            "duty_cycle": (0.0, 1.0),  # fraction of period
            "period": (0.001, 100.0),  # seconds
            "phase_1": (0.0, 2 * np.pi),  # radians
            "phase_2": (0.0, 2 * np.pi),  # radians
            "delay": (0.0, 100.0),  # milliseconds
        }

    def _resolve_params(self, params):
        params = {
            "c0": params.get("c0", self.normalize_param(self.c0, "c0")),
            "c1": params.get("c1", self.normalize_param(self.c1, "c1")),
            "c2": params.get("c2", self.normalize_param(self.c2, "c2")),
            "f1": params.get("f1", self.normalize_param(self.f1, "f1")),
            "f2": params.get("f2", self.normalize_param(self.f2, "f2")),
            "duty_cycle": params.get(
                "duty_cycle", self.normalize_param(self.duty_cycle, "duty_cycle")
            ),
            "period": params.get("period", self.normalize_param(self.period, "period")),
            "phase_1": params.get(
                "phase_1", self.normalize_param(self.phase_1, "phase_1")
            ),
            "phase_2": params.get(
                "phase_2", self.normalize_param(self.phase_2, "phase_2")
            ),
            "delay": params.get("delay", self.normalize_param(self.delay, "delay")),
        }

        unnormalized_params = {
            key: self.unnormalize_model_param(params[key], key) for key in params
        }
        return unnormalized_params

    def _is_active(self, t, params):
        period = params["period"]
        duty_cycle = params["duty_cycle"]
        return t >= 0 & ((t % period) < duty_cycle * period)

    def _compute_value(self, pulse_t, params):
        t = pulse_t
        c0 = params["c0"]
        c1 = params["c1"]
        c2 = params["c2"]
        f1 = params["f1"]
        f2 = params["f2"]
        phase_1 = params["phase_1"]
        phase_2 = params["phase_2"]

        return (
            c0
            + c1 * np.sin(2 * np.pi * f1 * t + phase_1)
            + c2 * np.sin(2 * np.pi * f2 * t + phase_2)
        )
