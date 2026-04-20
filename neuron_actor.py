import os

import numpy as np
import ray


def ensure_ray_initialized():
    """Initialize Ray exactly once per process.

    Prefers attaching to an existing local cluster (started by train.py or the
    user) so that all SubprocVecEnv workers share one head node. Falls back to
    starting a per-process cluster if no shared one is reachable.
    """
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    except (ConnectionError, Exception):
        ray.init(ignore_reinit_error=True, log_to_driver=False)


# num_cpus=0 keeps Ray from reserving logical CPU slots for these actors.
# NEURON is single-threaded per process but we routinely run more actors than
# cores (num_envs * len(neuron_types)), and the OS schedules them fine.
@ray.remote(num_cpus=0)
class NeuronActor:
    def __init__(
        self,
        cell_id,
        human_or_mice=0,
        temp=37.0,
        dt=0.1,
        sampling_rate=1e5,
        delay_init=2000,
        delay_final=5,
    ):
        os.environ.setdefault(
            "NEURON_MODULE_OPTIONS", "-nogui -NSTACK 100000 -NFRAME 20000"
        )
        from neuron_model_serial import NeuronSim

        self.cell_id = cell_id
        self.human_or_mice = human_or_mice
        self.temp = temp
        self.dt = dt
        self.sampling_rate = sampling_rate
        self.delay_init = int(delay_init)
        self.delay_final = int(delay_final)
        self.neuron = NeuronSim(
            human_or_mice=human_or_mice,
            cell_id=cell_id,
            temp=temp,
            dt=dt,
            elec_field=None,
            elec_field2=None,
        )

    def set_field(self, elec_field, pos_neuron=None, angle=None):
        if pos_neuron is None:
            pos_neuron = np.array([0, 0, 0])
        if angle is None:
            angle = np.array([0, 0])
        self.neuron._reset_elec_field(elec_field)
        self.neuron._set_xtra_param(
            pos_neuron=np.asarray(pos_neuron), angle=np.asarray(angle)
        )

        save_state = os.path.join(
            os.getcwd(),
            "cells/SaveState/human_or_mice"
            + str(self.human_or_mice)
            + "cell-"
            + str(self.cell_id)
            + "_Temp-"
            + str(float(self.temp))
            + "C_dt-"
            + str(self.dt * 1e3)
            + "us_delay-"
            + str(self.delay_init)
            + "ms.bin",
        )
        if not os.path.exists(save_state):
            n_samples = int(self.delay_init / (0.025 * 1e-3 * self.sampling_rate)) + 1
            time_array = np.arange(n_samples) / self.sampling_rate
            amp_array = np.zeros(n_samples)
            self.neuron.stimulate(
                time_array=time_array,
                amp_array=amp_array,
                amp_array2=amp_array.copy(),
                sampling_rate=self.sampling_rate,
                delay_init=self.delay_init,
                delay_final=self.delay_final,
                save_state_show=False,
            )
        return True

    def stimulate(self, waveform):
        waveform = np.asarray(waveform)
        time_array = np.arange(len(waveform)) / self.sampling_rate * 1000
        waveform2 = np.zeros_like(waveform)
        self.neuron.stimulate(
            time_array=time_array,
            amp_array=waveform,
            amp_array2=waveform2,
            scale1=1,
            sampling_rate=self.sampling_rate,
            delay_init=self.delay_init,
            delay_final=self.delay_final,
        )
        soma_recording, t = self.neuron.save_soma_recording(
            delay_init=self.delay_init
        )
        return np.asarray(soma_recording), np.asarray(t)
