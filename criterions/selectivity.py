from criterions.criterion import Criterion


class SelectivityCriterion(Criterion):
    # TODO: see how to implement several wavefors without breaking the interface
    def evaluate(self, waveform):
        # Placeholder for selectivity calculation
        selectivity = self._calculate_selectivity(waveform)
        return selectivity

    def _calculate_selectivity(self, waveform):
        # Implement the actual selectivity calculation logic here
        # This is a placeholder implementation
        return sum(waveform)  # Example: selectivity as sum of waveform values
        return sum(waveform)  # Example: selectivity as sum of waveform values
