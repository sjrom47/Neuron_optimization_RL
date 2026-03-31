from waveforms.waveform import Waveform


class FullSequenceWaveform(Waveform):
    def __init__(self, duration, sampling_rate):
        self.duration = duration
        self.sampling_rate = sampling_rate

    @property
    def n_params(self):
        return self.sampling_rate * self.duration  # full_sequence

    def _generate_waveform_points(self, duration, sampling_rate, params):
        return params.get("full_sequence", self.default_stimulation())

    def default_stimulation(self):
        pass
