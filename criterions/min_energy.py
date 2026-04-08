from criterions.criterion import Criterion


class MinEnergy(Criterion):
    def __init__(self):
        pass

    @property
    def requires_multiple_responses(self):
        return False

    def evaluate(self, response):
        if isinstance(response, list):
            response = response[0]  # Assuming single response for now
        # Placeholder for energy calculation
        spikes = super().calculate_n_spikes(response)
        has_spikes = int(spikes > 0)
        # TODO: factor length in to get actual energy values and not something proportional
        energy = sum(response**2)  # Example: energy as sum of squares
        # TODO: probably have to change this because it will produce flat waveforms for min energy
        # and it will reward hack. What matters is the magnitude of the energy
        return has_spikes - energy  # Negative energy for minimization
