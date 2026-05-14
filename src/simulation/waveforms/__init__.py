"""Waveform parameterizations available to the RL environment."""

from simulation.waveforms.charge_balanced import ChargeBalancedWaveform
from simulation.waveforms.fourier_approx import FourierWaveform
from simulation.waveforms.legendre3 import Legendre3Waveform
from simulation.waveforms.square import SquareWaveform
from simulation.waveforms.two_sines import TwoSinesWaveform

__all__ = [
    "ChargeBalancedWaveform",
    "FourierWaveform",
    "Legendre3Waveform",
    "SquareWaveform",
    "TwoSinesWaveform",
]
