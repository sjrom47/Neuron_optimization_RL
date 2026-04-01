import os
import random

import numpy as np
from gymnasium import Env, spaces

from elec_field import ICMS
from neuron_model_serial import NeuronSim
from waveforms import FourierWaveform, Legendre3Waveform, SquareWaveform


class NEURONEnv(Env):
    def __init__(self, waveform_type, criterion, max_actions=10, sampling_rate=1e6):
        super().__init__()
        self.waveform_type = waveform_type
        self.criterion = criterion
        self.waveform_type = waveform_type
        self.waveform = self.init_waveform()
        self.state = spaces.Dict(
            {
                "electrode_radius": spaces.Box(
                    low=0.5, high=1.5, shape=(), dtype=float
                ),  # TODO: change to actual bounds
                "theta": spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=float),
                "phi": spaces.Box(low=0.0, high=np.pi, shape=(), dtype=float),
                "neuron_type": spaces.Discrete(2),  # Assuming 2 neuron types
                # TODO: change actual bounds and types for these parameters
                "last_waveform_params": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
                ),
                # TODO: change to actual bounds and types for these parameters
                "last_stimulation_params": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=float
                ),
            }
        )
        self.actions_taken = 0
        self.max_actions = max_actions
        self.sampling_rate = sampling_rate

    def init_waveform(self, **kwargs):
        if self.waveform_type == "fourier":
            return FourierWaveform(**kwargs)
        elif self.waveform_type == "legendre3":
            return Legendre3Waveform(**kwargs)
        elif self.waveform_type == "square":
            return SquareWaveform(**kwargs)
        else:
            raise ValueError(f"Unsupported waveform type: {self.waveform_type}")

    def step(self, action):
        waveform, params = self.waveform.generate_waveform(
            duration=1.0, sampling_rate=self.sampling_rate, params=action
        )
        response = self.simulate_neuron_response(waveform)
        reward = self.criterion.evaluate(response)

        # TODO: update state based on response and/or action

        self.actions_taken += 1
        terminated = self.actions_taken >= self.max_actions
        truncated = False  # You can implement truncation logic if needed

        return self.state, reward, terminated, truncated, {}

    def simulate_neuron_response(self, waveform):
        time_array = np.arange(len(waveform)) / self.sampling_rate
        waveform2 = np.zeros_like(waveform)

        self.neuron.stimulate(
            time_array=time_array,
            waveform=waveform,
            waveform2=waveform2,
            scale1=1,
            sampling_rate=self.sampling_rate,
            delay_init=2000,
            delay_final=5,
        )
        soma_recording, _ = self.neuron.save_soma_recording(delay_init=2000)
        return soma_recording

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.state["electrode_radius"] = random.uniform(0.5, 1.5)
        self.state["theta"] = random.uniform(0.0, 2 * np.pi)
        self.state["phi"] = random.uniform(0.0, np.pi)

        # TODO: fix this
        self.state["neuron_type"] = random.choice(
            list(range(self.state["neuron_type"].n))
        )  # Neuron type

        r = float(self.state["electrode_radius"])
        theta = float(self.state["theta"])
        phi = float(self.state["phi"])
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        elec_field = ICMS(x=x, y=y, z=z, conductivity=0.33)

        self.neuron = NeuronSim(
            human_or_mice=0,
            cell_id=self.state["neuron_type"],
            temp=37,
            dt=0.025,
            elec_field=elec_field,
            elec_field2=None,
        )
        self.neuron._set_xtra_param(
            angle=np.array([0, 0]), pos_neuron=np.array([0, 0, 0])
        )

        delay_init, delay_final = 2000, 5
        save_state = os.path.join(
            os.getcwd(),
            "cells/SaveState/human_or_mice0cell-"
            + str(self.state["neuron_type"])
            + "_Temp-37C_dt-25.0us_delay-2000ms.bin",
        )
        if not os.path.exists(save_state):
            n_samples = int(delay_init / (0.025 * 1e-3 * self.sampling_rate)) + 1
            time_array = np.arange(n_samples) / self.sampling_rate
            amp_array = np.zeros(n_samples)
            self.neuron.stimulate(
                time_array=time_array,
                amp_array=amp_array,
                amp_array2=amp_array.copy(),
                sampling_rate=self.sampling_rate,
                delay_init=delay_init,
                delay_final=delay_final,
                save_state_show=False,
            )

        default_waveform = self.waveform.default_stimulation()
        response = self.simulate_neuron_response(default_waveform)

        # TODO: transform response into something useful for the agent (e.g. firing rate, latency, etc.)
        # TODO: Add that info into the state representation

        # TODO: Add stimulation parameters into the state representation
        # TODO: see how to handle variable number of parameters for different waveforms (Fourier has more parameters than square for example)

        return self.state
