import numpy as np

from simulation.criteria.criterion import Criterion


class MinEnergy(Criterion):
    def __init__(self, max_amplitude=500.0, energy_weight=5, spike_threshold_vm=0.0):
        self.max_amplitude = float(max_amplitude)
        self._lambda = float(energy_weight)
        self._spike_threshold_vm = float(spike_threshold_vm)

    @property
    def requires_multiple_responses(self):
        return False

    def evaluate(self, waveform, response):
        if isinstance(response, list):
            response = response[0]
        spikes = super().calculate_n_spikes(response)

        # Energy normalized to [0, 1] so the penalty is consistent across waveform
        # types regardless of max_amplitude.
        energy_frac = float(np.mean(waveform**2) / (self.max_amplitude**2))

        # Fractional approach to spike threshold. response[0] is Vm before stimulation,
        # a data-derived proxy for resting potential.
        resting = float(response[0])
        peak_vm = float(np.max(response))
        denom = max(self._spike_threshold_vm - resting, 1e-6)
        depol = float(np.clip((peak_vm - resting) / denom, 0.0, 1.0))

        # Do not apply energy penalty if there are no spikes or the flat
        # waveform becomes a local optimum. This makes optimization much harder
        energy_penalty = self._lambda * energy_frac if spikes > 0 else 0.0

        # Scale depol by 1/lambda so its maximum (1/lambda) is always below the
        # benefit of one spike at the break-even energy level (energy_frac=1/lambda).
        # Without this, near-threshold depolarization without spiking is better
        # than actually spiking.
        depol_weight = 1.0 / max(self._lambda, 1.0)
        return spikes + depol_weight * depol - energy_penalty
