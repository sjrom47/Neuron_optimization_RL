import numpy as np

from criterions.criterion import Criterion


class SelectivityCriterion(Criterion):
    def __init__(self, max_amplitude=1000.0, energy_weight=10):
        self.max_amplitude = float(max_amplitude)
        self._lambda = float(energy_weight)

    @property
    def requires_multiple_responses(self):
        return True

    def evaluate(self, waveform, responses):
        target = responses[0]
        target_spikes = self.calculate_n_spikes(target)
        other_spikes = sum(self.calculate_n_spikes(r) for r in responses[1:])
        selectivity = target_spikes - other_spikes

        # Energy normalized to [0, 1] so the penalty is consistent across
        # waveform types regardless of max_amplitude.
        energy_frac = float(np.mean(waveform**2) / (self.max_amplitude**2))

        # Gate the energy penalty on the target actually spiking. Otherwise
        # the trivial zero-amplitude policy is a strong local optimum the
        # agent cannot escape: exploring toward larger amplitudes pays the
        # energy cost before discovering target spikes.
        energy_penalty = self._lambda * energy_frac if target_spikes > 0 else 0.0

        return selectivity - energy_penalty
