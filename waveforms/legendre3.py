import numpy as np

from waveforms.waveform import Waveform


class Legendre3Waveform(Waveform):
    def __init__(self, c0, c1, c2, frequency=1.0, duty_cycle=0.5, delay=0.0):
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.delay = delay

    def _generate_waveform_points(self, duration, sampling_rate, params):
        # TODO
        return super()._generate_waveform_points(duration, sampling_rate, params)

    def convert_to_interval_bounds(self, t_points, active_points):
        # Calculate the period of the waveform
        pass
