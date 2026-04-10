import os
import random

os.environ["NEURON_MODULE_OPTIONS"] = "-nogui -NSTACK 100000 -NFRAME 20000"

import time

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, spaces
from neuron import coreneuron

coreneuron.enable = True
coreneuron.gpu = True

from criterions import MinEnergy, SelectivityCriterion
from elec_field import ICMS
from neuron_model_serial import NeuronSim
from utils import firing_rate
from waveforms import FourierWaveform, Legendre3Waveform, SquareWaveform


class NEURONEnv(Env):
    def __init__(
        self, waveform_type, criterion_type, max_actions=10, sampling_rate=1e5
    ):
        super().__init__()

        # self.neuron_types = [6, 7, 35, 36]
        self.neuron_types = [36]

        self.criterion = self.init_criterion(criterion_type)
        # TODO: we will have to actually see if we use the 4 neuron types or we decide to use more
        self.n_neuron_state_params = (
            3 * len(self.neuron_types)
            if self.criterion.requires_multiple_responses
            else 3
        )
        self.best_reward = -np.inf

        self.waveform = self.init_waveform(waveform_type)
        total_dim = (
            1  # electrode_radius
            + 1  # theta
            + 1  # phi
            + 1  # neuron_type
            + self.waveform.n_params  # last_waveform_params
            + self.n_neuron_state_params  # last_stimulation_params
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float64
        )

        # Separate plain dict to hold actual observation values
        self.state = {
            "electrode_radius": np.float64(0.0),
            "theta": np.float64(0.0),
            "phi": np.float64(0.0),
            "neuron_type": 0,
            # TODO: change to actual bounds and types for these parameters
            # also consider multi neuron criterions
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
        self.stimulation_duration = 30  # ms
        self.delay_init = 2000  # ms
        self.delay_final = 5  # ms

    def get_obs(self):
        return np.concatenate(
            [
                [self.state["electrode_radius"]],
                [self.state["theta"]],
                [self.state["phi"]],
                [float(self.state["neuron_type"])],
                self.state["last_waveform_params"],
                self.state["last_stimulation_params"],
            ]
        ).astype(np.float64)

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
        times = []
        if self.criterion.requires_multiple_responses:
            for neuron_type in self.neuron_types:
                response, t = self.simulate_neuron_response(waveform, neuron_type)
                responses.append(response)
                times.append(t)
        else:
            response, t = self.simulate_neuron_response(
                waveform, self.state["neuron_type"]
            )
            responses.append(response)
            times.append(t)
        return responses, times

    def step(self, action):
        t0 = time.time()
        action_dict = {
            key: action[i] for i, key in enumerate(self.waveform.param_bounds.keys())
        }

        waveform, params = self.waveform.generate_waveform(
            duration=self.stimulation_duration,
            sampling_rate=self.sampling_rate,
            params=action_dict,
        )

        responses, times = self.get_neuron_responses(waveform)

        reward = self.criterion.evaluate(waveform, responses)

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
        t1 = time.time()
        unnormalized_params = {
            key: self.waveform.unnormalize_model_param(action[i], key)
            for i, key in enumerate(self.waveform.param_bounds.keys())
        }
        print(f"Step time: {t1 - t0}, Reward: {reward}")
        if reward > self.best_reward:
            self.best_reward = reward
            self.plot_waveform_and_response(
                waveform, responses[0], times[0], unnormalized_params, reward=reward
            )
        if terminated:
            self.plot_waveform_and_response(
                waveform,
                responses[0],
                times[0],
                unnormalized_params,
                plot_name="terminated",
                reward=reward,
            )
        return self.get_obs(), reward, terminated, truncated, {}

    def plot_waveform_and_response(
        self,
        waveform,
        response,
        time_response,
        params,
        plot_name="best_response",
        reward=None,
    ):
        if reward is None:
            reward = self.best_reward
        os.makedirs("plots", exist_ok=True)
        t_waveform = np.arange(len(waveform)) / self.sampling_rate * 1000  # ms

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(
            f"Params: {' '.join([f'{key}={params[key]:.2f}' for key in params])}\n Reward: {reward:.3f}"
        )
        axs[0].plot(t_waveform, waveform)
        axs[0].set_title("Stimulation Waveform")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_ylim(
            -1.1 * self.waveform.max_amplitude, 1.1 * self.waveform.max_amplitude
        )

        axs[1].plot(time_response, response)
        axs[1].set_title("Neuron Response")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Voltage (mV)")
        axs[1].set_ylim(-100, 40)

        plt.tight_layout()
        plt.savefig(f"plots/{plot_name}.png")
        plt.close()

    def simulate_neuron_response(self, waveform, neuron_type):
        time_array = (
            np.arange(len(waveform)) / self.sampling_rate * 1000
        )  # convert to ms
        waveform2 = np.zeros_like(waveform)

        start_time = time.time()
        self.neurons[neuron_type].stimulate(
            time_array=time_array,
            amp_array=waveform,
            amp_array2=waveform2,
            scale1=1,
            sampling_rate=self.sampling_rate,
            delay_init=self.delay_init,
            delay_final=self.delay_final,
        )
        mid_time = time.time()
        soma_recording, t_neuron = self.neurons[neuron_type].save_soma_recording(
            delay_init=self.delay_init
        )
        end_time = time.time()
        print(
            f"Neuron simulation time: {end_time - start_time}, recording time: {end_time - mid_time}"
        )

        return soma_recording, t_neuron

    def get_stimulation_params(self, response):
        fr = firing_rate(response, np.arange(len(response)) / self.sampling_rate)
        peak_vm = np.max(response)
        last_mp = response[-1]
        return fr, peak_vm, last_mp

    def default_stimulation(self):
        default_params = {key: 0.0 for key in self.waveform.param_bounds.keys()}
        default_waveform, _ = self.waveform.generate_waveform(
            duration=self.stimulation_duration,
            sampling_rate=self.sampling_rate,
            params=default_params,
        )
        return default_waveform

    def init_neuron(self, neuron_type, elec_field):
        default_waveform = self.default_stimulation()

        time_array = np.arange(len(default_waveform)) / self.sampling_rate
        neuron = NeuronSim(
            human_or_mice=0,
            cell_id=neuron_type,
            temp=37.0,
            dt=0.1,
            elec_field=elec_field,
            elec_field2=None,
        )
        neuron._set_xtra_param(angle=np.array([0, 0]), pos_neuron=np.array([0, 0, 0]))

        delay_init, delay_final = self.delay_init, self.delay_final
        save_state = os.path.join(
            os.getcwd(),
            "cells/SaveState/human_or_mice0cell-"
            + str(neuron_type)
            + "_Temp-37.0C_dt-100.0us_delay-"
            + str(delay_init)
            + "ms.bin",
        )
        print(f"Checking for saved state at: {save_state}")
        if not os.path.exists(save_state):
            print("Saved state not found")
            n_samples = int(delay_init / (0.025 * 1e-3 * self.sampling_rate)) + 1
            time_array = np.arange(n_samples) / self.sampling_rate
            amp_array = np.zeros(n_samples)
            neuron.stimulate(
                time_array=time_array,
                amp_array=amp_array,
                amp_array2=amp_array.copy(),
                sampling_rate=self.sampling_rate,
                delay_init=delay_init,
                delay_final=delay_final,
                save_state_show=False,
            )
        return neuron

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.actions_taken = 0

        self.state["electrode_radius"] = np.float64(1)
        self.state["theta"] = np.float64(0)
        self.state["phi"] = np.float64(0)
        # self.state["electrode_radius"] = np.float64(random.uniform(0.5, 1.5))
        # self.state["theta"] = np.float64(random.uniform(0.0, 2 * np.pi))
        # self.state["phi"] = np.float64(random.uniform(0.0, np.pi))
        self.state["neuron_type"] = int(random.choice(self.neuron_types))

        r = float(self.state["electrode_radius"])
        theta = float(self.state["theta"])
        phi = float(self.state["phi"])
        x = 0
        y = 1
        z = 0
        # x = r * np.sin(phi) * np.cos(theta)
        # y = r * np.sin(phi) * np.sin(theta)
        # z = r * np.cos(phi)

        default_waveform = self.default_stimulation()

        self.elec_field = ICMS(x=x, y=y, z=z, conductivity=0.33)
        self.neurons = {}
        self.neurons[self.state["neuron_type"]] = self.init_neuron(
            self.state["neuron_type"], self.elec_field
        )
        if self.criterion.requires_multiple_responses:
            for neuron_type in self.neuron_types:
                if neuron_type == self.state["neuron_type"]:
                    continue  # Skip the already initialized neuron for the selected neuron type
                neuron = self.init_neuron(neuron_type, self.elec_field)
                self.neurons[neuron_type] = neuron

        response, _ = self.simulate_neuron_response(
            default_waveform, self.state["neuron_type"]
        )

        neuron_stimulation_params = []
        base_neuron_stimulation_params = self.get_stimulation_params(response)
        neuron_stimulation_params.extend(base_neuron_stimulation_params)

        if self.criterion.requires_multiple_responses:
            for neuron_type in self.neuron_types:
                if neuron_type == self.state["neuron_type"]:
                    continue  # Skip the already computed response for the selected neuron type
                response, _ = self.simulate_neuron_response(
                    default_waveform, neuron_type
                )
                params = self.get_stimulation_params(response)
                neuron_stimulation_params.extend(params)
        self.state["last_stimulation_params"] = np.array(neuron_stimulation_params)

        self.state["last_waveform_params"] = np.zeros(
            self.waveform.n_params
        )  # No previous waveform parameters at reset

        return self.get_obs(), {}
