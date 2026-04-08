import os
import random

import numpy as np
from gymnasium import Env, spaces

from criterions import MinEnergy, SelectivityCriterion
from elec_field import ICMS
from neuron_model_serial import NeuronSim
from utils import firing_rate
from waveforms import FourierWaveform, Legendre3Waveform, SquareWaveform


class NEURONEnv(Env):
    def __init__(
        self, waveform_type, criterion_type, max_actions=10, sampling_rate=1e6
    ):
        super().__init__()

        self.neuron_types = [6, 7, 35, 36]

        self.criterion = self.init_criterion(criterion_type)
        # TODO: we will have to actually see if we use the 4 neuron types or we decide to use more
        self.n_neuron_state_params = (
            3 * len(self.neuron_types)
            if self.criterion.requires_multiple_responses
            else 3
        )

        self.waveform = self.init_waveform(waveform_type)
        # self.state = spaces.Dict(
        #     {
        #         "electrode_radius": spaces.Box(
        #             low=0.5, high=1.5, shape=(), dtype=float
        #         ),
        #         "theta": spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=float),
        #         "phi": spaces.Box(low=0.0, high=np.pi, shape=(), dtype=float),
        #         "neuron_type": spaces.Discrete(4),  # Assuming 4 neuron types
        #         "last_waveform_params": spaces.Box(
        #             low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
        #         ),
        #         "last_stimulation_params": spaces.Box(
        #             low=-1e6, high=1e6, shape=(self.n_neuron_state_params,), dtype=float
        #         ),
        #     }
        # )

        self.observation_space = spaces.Dict(
            {
                "electrode_radius": spaces.Box(
                    low=0.5, high=1.5, shape=(), dtype=float
                ),
                "theta": spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=float),
                "phi": spaces.Box(low=0.0, high=np.pi, shape=(), dtype=float),
                "neuron_type": spaces.Discrete(4),
                "last_waveform_params": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
                ),
                # TODO: change to actual bounds and types for these parameters
                # also consider multi neuron criterions
                "last_stimulation_params": spaces.Box(
                    low=-1e6, high=1e6, shape=(self.n_neuron_state_params,), dtype=float
                ),
            }
        )

        # Separate plain dict to hold actual observation values
        self.state = {
            "electrode_radius": np.float64(0.0),
            "theta": np.float64(0.0),
            "phi": np.float64(0.0),
            "neuron_type": 0,
            "last_waveform_params": np.zeros(self.waveform.n_params, dtype=float),
            "last_stimulation_params": np.zeros(
                self.n_neuron_state_params, dtype=float
            ),
        }
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
        )

        self.actions_taken = 0
        self.max_actions = max_actions
        self.sampling_rate = sampling_rate

    def init_waveform(self, waveform_type, **kwargs):
        if waveform_type == "fourier":
            return FourierWaveform(**kwargs)
        elif waveform_type == "legendre3":
            return Legendre3Waveform(**kwargs)
        elif waveform_type == "square":
            return SquareWaveform(**kwargs)
        else:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")

    def init_criterion(self, criterion_type, **kwargs):
        if criterion_type == "min_energy":
            return MinEnergy(**kwargs)
        elif criterion_type == "selectivity":
            return SelectivityCriterion(**kwargs)
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def get_neuron_responses(self, waveform):
        responses = []
        if self.criterion.requires_multiple_responses:
            for neuron_type in self.neuron_types:
                response = self.simulate_neuron_response(waveform, neuron_type)
                responses.append(response)
        else:
            response = self.simulate_neuron_response(
                waveform, self.state["neuron_type"]
            )
            responses.append(response)
        return responses

    def step(self, action):
        waveform, params = self.waveform.generate_waveform(
            duration=1.0, sampling_rate=self.sampling_rate, params=action
        )
        responses = self.get_neuron_responses(waveform)
        reward = self.criterion.evaluate(responses)

        self.state["last_waveform_params"] = np.array(
            [params[key] for key in self.waveform.param_bounds.keys()]
        )

        stimulation_params = self.get_stimulation_params(
            responses[0]
        )  # Get params for the first response
        self.state["last_stimulation_params"] = np.array(stimulation_params)

        self.actions_taken += 1
        terminated = self.actions_taken >= self.max_actions
        truncated = False  # You can implement truncation logic if needed

        return self.state, reward, terminated, truncated, {}

    def simulate_neuron_response(self, waveform, neuron_type):
        time_array = np.arange(len(waveform)) / self.sampling_rate
        waveform2 = np.zeros_like(waveform)
        # TODO: for future improvements, consider creating the neurons in reset and
        # just using them here instead of creating a new one every time
        # Also figure out exactly how to make it different for the transfer learning part
        self.neuron = NeuronSim(
            human_or_mice=0,
            cell_id=neuron_type,
            temp=37,
            dt=0.025,
            elec_field=self.elec_field,
            elec_field2=None,
        )
        self.neuron._set_xtra_param(
            angle=np.array([0, 0]), pos_neuron=np.array([0, 0, 0])
        )

        delay_init, delay_final = 2000, 5
        save_state = os.path.join(
            os.getcwd(),
            "cells/SaveState/human_or_mice0cell-"
            + str(neuron_type)
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

        self.neuron.stimulate(
            time_array=time_array,
            amp_array=waveform,
            amp_array2=waveform2,
            scale1=1,
            sampling_rate=self.sampling_rate,
            delay_init=2000,
            delay_final=5,
        )
        soma_recording, _ = self.neuron.save_soma_recording(delay_init=2000)
        return soma_recording

    def get_stimulation_params(self, response):
        fr = firing_rate(response, np.arange(len(response)) / self.sampling_rate)
        peak_vm = np.max(response)
        last_mp = response[-1]
        return fr, peak_vm, last_mp

    def default_stimulation(self):
        default_params = {key: 0.0 for key in self.waveform.param_bounds.keys()}
        default_waveform, _ = self.waveform.generate_waveform(
            duration=1.0, sampling_rate=self.sampling_rate, params=default_params
        )
        return default_waveform

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.actions_taken = 0

        self.state["electrode_radius"] = np.float64(random.uniform(0.5, 1.5))
        self.state["theta"] = np.float64(random.uniform(0.0, 2 * np.pi))
        self.state["phi"] = np.float64(random.uniform(0.0, np.pi))
        self.state["neuron_type"] = int(random.choice(self.neuron_types))

        r = float(self.state["electrode_radius"])
        theta = float(self.state["theta"])
        phi = float(self.state["phi"])
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        self.elec_field = ICMS(x=x, y=y, z=z, conductivity=0.33)

        default_waveform = self.default_stimulation()
        response = self.simulate_neuron_response(
            default_waveform, self.state["neuron_type"]
        )

        neuron_stimulation_params = []
        base_neuron_stimulation_params = self.get_stimulation_params(response)
        neuron_stimulation_params.extend(base_neuron_stimulation_params)

        if self.criterion.requires_multiple_responses:
            for neuron_type in self.neuron_types:
                if neuron_type == self.state["neuron_type"]:
                    continue  # Skip the already computed response for the selected neuron type
                response = self.simulate_neuron_response(default_waveform, neuron_type)
                params = self.get_stimulation_params(response)
                neuron_stimulation_params.extend(params)
        self.state["last_stimulation_params"] = np.array(neuron_stimulation_params)

        self.state["last_waveform_params"] = np.zeros(
            self.waveform.n_params
        )  # No previous waveform parameters at reset

        return self.state, {}
