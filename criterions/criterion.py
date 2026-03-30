from abc import ABC, abstractmethod


class Criterion(ABC):
    @abstractmethod
    def evaluate(self, waveform):
        pass

    def n_spikes(self, waveform):
        # TODO: implement the logic for n_peaks using HW1 code
        pass
