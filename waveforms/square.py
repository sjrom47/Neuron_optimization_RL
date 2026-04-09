import numpy as np

from waveforms.waveform import Waveform


class SquareWaveform(Waveform):
    def __init__(
        self,
        period=10.0,
        amplitude=100.0,
        duty_cycle=1.0,
        delay=0.0,
        max_amplitude=500.0,
    ):
        super().__init__(max_amplitude)
        self.period = period
        self.amplitude = min(amplitude, max_amplitude)
        # Changed from pulse width to duty cycle because as a fraction of the period
        # it is easier to handle constraints
        self.duty_cycle = duty_cycle
        self.delay = delay

    @property
    def n_params(self):
        return 4  # frequency, amplitude, duty_cycle, delay

    @property
    def param_bounds(self):
        # TODO: change to actual bounds for these parameters
        return {
            "period": (0.0, 100.0),  # milliseconds
            "amplitude": (0.0, self.max_amplitude),  # mA
            "duty_cycle": (0.0, 1.0),  # fraction of period
            "delay": (0.0, 100.0),  # milliseconds
        }

    def _resolve_params(self, params):
        amplitude = params.get("amplitude", self.amplitude)
        duty_cycle = params.get("duty_cycle", self.duty_cycle)
        delay = params.get("delay", self.delay)
        period = params.get("period", self.period)

        return {
            "amplitude": amplitude,
            "duty_cycle": duty_cycle,
            "delay": delay,
            "period": period,
        }

    def _is_active(self, t, params):
        period = params["period"]
        duty_cycle = params["duty_cycle"]
        return (t >= 0) & ((t % period) < duty_cycle * period)

    def _compute_value(self, pulse_t, params):
        return np.full(len(pulse_t), params["amplitude"])
