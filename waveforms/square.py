import numpy as np

from waveforms.waveform import Waveform


class SquareWaveform(Waveform):
    def __init__(
        self,
        frequency=100.0,
        amplitude=100.0,
        duty_cycle=1.0,
        delay=0.0,
        max_amplitude=2000.0,
    ):
        super().__init__(max_amplitude)
        self.frequency = frequency
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
            "frequency": (0.001, 1000.0),  # Hz
            "amplitude": (0.0, 2000.0),  # mA
            "duty_cycle": (0.0, 1.0),  # fraction of period
            "delay": (0.0, 100.0),  # milliseconds
        }

    def _resolve_params(self, params):
        frequency = params.get("frequency", self.frequency)
        amplitude = params.get("amplitude", self.amplitude)
        duty_cycle = params.get("duty_cycle", self.duty_cycle)
        delay = params.get("delay", self.delay)
        period = 1.0 / frequency

        return {
            "frequency": frequency,
            "amplitude": amplitude,
            "duty_cycle": duty_cycle,
            "delay": delay,
            "period": period,
        }

    def _is_active(self, t, params):
        period = params["period"]
        duty_cycle = params["duty_cycle"]
        return t >= 0 and (t % period) < duty_cycle * period

    def _compute_value(self, pulse_t, params):
        return [params["amplitude"] * len(pulse_t)]
