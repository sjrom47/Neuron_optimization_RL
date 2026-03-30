import numpy as np

from waveforms.waveform import Waveform


class SquareWaveform(Waveform):
    def __init__(self, frequency, amplitude=1.0, pulse_width=1.0, delay=0.0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        self.delay = delay

    @property
    def n_params(self):
        return 4  # frequency, amplitude, pulse_width, delay

    def _generate_waveform_points(self, duration, sampling_rate, params):
        # TODO:
        t_points = self.get_t_points(duration, sampling_rate)
        waveform = np.zeros_like(t_points)

        # Calculate the period of the square wave
        period = 1.0 / self.frequency

        # TODO: finish doing this
        t_points -= self.delay  # Shift time points by the delay
        waveform[t_points < self.pulse_width] = (
            self.amplitude
        )  # Set amplitude for points within pulse width

        return waveform
