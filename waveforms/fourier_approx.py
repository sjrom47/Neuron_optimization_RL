import numpy as np

from waveforms.waveform import Waveform


class FourierWaveform(Waveform):
    def __init__(
        self,
        c0,
        c1,
        c2,
        f1,
        f2,
        duty_cycle=0.5,
        period=1.0,
        phase_1=0.0,
        phase_2=0.0,
        delay=0.0,
    ):
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.f1 = f1
        self.f2 = f2
        self.duty_cycle = duty_cycle
        self.period = period
        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.delay = delay

    def default_stimulation(self):
        return self.generate_waveform(duration=1.0, sampling_rate=1000)

    def _generate_waveform_points(self, duration, sampling_rate, params):
        c0 = params.get("c0", self.c0)
        c1 = params.get("c1", self.c1)
        c2 = params.get("c2", self.c2)
        f1 = params.get("f1", self.f1)
        f2 = params.get("f2", self.f2)
        phase_1 = params.get("phase_1", self.phase_1)
        phase_2 = params.get("phase_2", self.phase_2)
        duty_cycle = params.get("duty_cycle", self.duty_cycle)
        period = params.get("period", self.period)
        delay = params.get("delay", self.delay)

        t_points = self.get_t_points(duration, sampling_rate)
        # waveform_points = c0 + c1 * np.sin(2 * np.pi * f1 * t_points + phase_1) + c2 * np.sin(2 * np.pi * f2 * t_points + phase_2)
        # TODO: apply duty cycle and period
