from waveforms.waveform import Waveform


class FullSequenceWaveform(Waveform):
    def __init__(self, duration, sampling_rate, max_amplitude=2000.0):
        super().__init__(max_amplitude)
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.max_amplitude = max_amplitude

    @property
    def n_params(self):
        return self.sampling_rate * self.duration  # full_sequence

    @property
    def param_bounds(self):
        return {
            "full_sequence": [(-self.max_amplitude, self.max_amplitude)]
            * int(self.sampling_rate * self.duration)
        }

    def _is_active(self, t, params):
        return True

    def _compute_value(self, pulse_t, params):
        return params["full_sequence"]

    def _resolve_params(self, params):
        return {
            "full_sequence": params.get(
                "full_sequence", [0.0] * int(self.sampling_rate * self.duration)
            )
        }
