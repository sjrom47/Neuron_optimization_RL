import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ray
from gymnasium import Env, spaces

from config import (
    DELAY_FINAL,
    DELAY_INIT,
    ELECTRODE_POSITION_PERTURBATION_SIGMA,
    NEURON_DT,
    NEURON_HUMAN_OR_MICE,
    NEURON_TEMP,
    NEURON_TYPES,
    PERTURBATE_ELECTRODE_POSITION,
    SAMPLING_RATE,
    STIMULATION_DURATION,
    MULTPLE_NEURON_TYPES,
)
from criterions import MinEnergy, SelectivityCriterion
from elec_field import ICMS
from neuron_actor import NeuronActor, ensure_ray_initialized
from utils import firing_rate
from waveforms import (
    ChargeBalancedWaveform,
    FourierWaveform,
    Legendre3Waveform,
    SquareWaveform,
    TwoSinesWaveform,
)


class NEURONEnv(Env):
    def __init__(
        self,
        waveform_type,
        criterion_type,
        max_actions=10,
        sampling_rate=SAMPLING_RATE,
        max_amplitude=1000.0,
    ):
        super().__init__()

        self.neuron_types = NEURON_TYPES

        self.waveform = self.init_waveform(waveform_type, max_amplitude=max_amplitude)
        self.criterion = self.init_criterion(
            criterion_type, max_amplitude=self.waveform.max_amplitude
        )
        # TODO: we will have to actually see if we use the 4 neuron types or we decide to use more
        self.n_neuron_state_params = (
            3 * len(self.neuron_types)
            if self.criterion.requires_multiple_responses
            else 3
        )
        total_dim = (
            1  # electrode_radius
            + 1  # theta
            + 1  # phi
            + 1  # neuron_type
            + self.waveform.n_params  # last_waveform_params
            + self.n_neuron_state_params  # last_stimulation_params
            + self.waveform.n_params  # best_waveform_params
            + 1  # best_reward
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
            "last_waveform_params": np.zeros(self.waveform.n_params, dtype=float),
            "last_stimulation_params": np.zeros(
                self.n_neuron_state_params, dtype=float
            ),
            "best_waveform_params": np.zeros(self.waveform.n_params, dtype=float),
            "best_reward": np.float64(0.0),
        }
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
        )

        self.actions_taken = 0
        self.max_actions = max_actions
        self.sampling_rate = sampling_rate
        self.stimulation_duration = STIMULATION_DURATION
        self.delay_init = DELAY_INIT
        self.delay_final = DELAY_FINAL
        self.vm_min = -100.0
        self.vm_max = 40.0
        self.fr_tanh_scale = 300.0

        # Each neuron type gets its own Ray actor holding a single NEURON
        # instance. Cant run two neuron types in one process
        ensure_ray_initialized()
        self.actors = {
            nt: NeuronActor.remote(
                cell_id=nt,
                human_or_mice=NEURON_HUMAN_OR_MICE,
                temp=NEURON_TEMP,
                dt=NEURON_DT,
                sampling_rate=self.sampling_rate,
                delay_init=self.delay_init,
                delay_final=self.delay_final,
            )
            for nt in self.neuron_types
        }

    def get_obs(self):
        return np.concatenate(
            [
                [self.state["electrode_radius"]],
                [self.state["theta"]],
                [self.state["phi"]],
                [float(self.state["neuron_type"])],
                self.state["last_waveform_params"],
                self.state["last_stimulation_params"],
                self.state["best_waveform_params"],
                [self.state["best_reward"]],
            ]
        ).astype(np.float64)

    def init_waveform(self, waveform_type, **kwargs):
        if waveform_type == "fourier":
            return FourierWaveform(**kwargs)
        elif waveform_type == "legendre3":
            return Legendre3Waveform(**kwargs)
        elif waveform_type == "square":
            return SquareWaveform(**kwargs)
        elif waveform_type == "two_sines":
            return TwoSinesWaveform(**kwargs)
        elif waveform_type == "charge_balanced":
            return ChargeBalancedWaveform(**kwargs)
        else:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")

    def init_criterion(self, criterion_type, max_amplitude=1000.0, **kwargs):
        if criterion_type == "min_energy":
            return MinEnergy(max_amplitude=max_amplitude, **kwargs)
        elif criterion_type == "selectivity":
            return SelectivityCriterion(max_amplitude=max_amplitude, **kwargs)
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def _active_neuron_types(self):
        if self.criterion.requires_multiple_responses:
            return list(self.neuron_types)
        return [int(self.state["neuron_type"])]

    def get_neuron_responses(self, waveform):
        types = self._active_neuron_types()
        # Put the waveform in the object store once, then fan out to each
        # actor so they simulate concurrently. ray.get collects in order.
        wf_ref = ray.put(np.asarray(waveform))
        futures = [self.actors[nt].stimulate.remote(wf_ref) for nt in types]
        results = ray.get(futures)
        responses = [r[0] for r in results]
        times = [r[1] for r in results]
        return responses, times

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        learnable_keys = list(self.waveform.param_bounds.keys())[
            : self.waveform.n_params
        ]
        action_dict = {key: action[i] for i, key in enumerate(learnable_keys)}

        waveform, params = self.waveform.generate_waveform(
            duration=self.stimulation_duration,
            sampling_rate=self.sampling_rate,
            params=action_dict,
        )

        responses, times = self.get_neuron_responses(waveform)

        reward = self.criterion.evaluate(waveform, responses)

        last_waveform_params = np.array([params[key] for key in learnable_keys])
        self.state["last_waveform_params"] = last_waveform_params

        if reward > self.state["best_reward"]:
            self.state["best_reward"] = np.float64(reward)
            self.state["best_waveform_params"] = last_waveform_params.copy()

        all_stimulation_params = []
        for r in responses:
            all_stimulation_params.extend(self.get_stimulation_params(r))
        stimulation_params = all_stimulation_params[:3]
        self.state["last_stimulation_params"] = np.array(all_stimulation_params)
        spikes = int(self.criterion.calculate_n_spikes(responses[0]))
        state_fr = float(stimulation_params[0])
        state_peak_vm = float(stimulation_params[1])
        state_last_mp = float(stimulation_params[2])

        self.actions_taken += 1
        # Reaching max_actions is a time limit, not a terminal state. Using
        # truncated (not terminated) lets SB3 bootstrap V(s') at the boundary
        # instead of zeroing it, which is the right target for this problem.
        terminated = False
        truncated = self.actions_taken >= self.max_actions
        unnormalized_params = {
            key: self.waveform.unnormalize_model_param(action[i], key)
            for i, key in enumerate(learnable_keys)
        }
        info = {
            "waveform": waveform,
            "response": responses[0],
            "time_response": times[0],
            "all_responses": responses,
            "all_times": times,
            "neuron_types": (
                list(self.neuron_types)
                if self.criterion.requires_multiple_responses
                else None
            ),
            "params": unnormalized_params,
            "reward": reward,
            "spikes": spikes,
            "state_fr": state_fr,
            "state_peak_vm": state_peak_vm,
            "state_last_mp": state_last_mp,
            "electrode_radius": float(self.state["electrode_radius"]),
            "theta": float(self.state["theta"]),
            "phi": float(self.state["phi"]),
            "terminated": terminated,
            "max_amplitude": self.waveform.max_amplitude,
            "sampling_rate": self.sampling_rate,
        }
        return self.get_obs(), reward, terminated, truncated, info

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

    def get_stimulation_params(self, response):
        fr_raw = firing_rate(response, np.arange(len(response)) / self.sampling_rate)
        fr = self._normalize_firing_rate(fr_raw)
        peak_vm = self._normalize_voltage(np.max(response))
        last_mp = self._normalize_voltage(response[-1])
        return fr, peak_vm, last_mp

    def _normalize_firing_rate(self, fr):
        scale = max(float(self.fr_tanh_scale), 1e-9)
        return float(np.tanh(float(fr) / scale))

    def _normalize_voltage(self, vm):
        vm = float(np.clip(vm, self.vm_min, self.vm_max))
        return 2.0 * (vm - self.vm_min) / (self.vm_max - self.vm_min) - 1.0

    def default_stimulation(self):
        default_params = {key: 0.0 for key in self.waveform.param_bounds.keys()}
        default_waveform, _ = self.waveform.generate_waveform(
            duration=self.stimulation_duration,
            sampling_rate=self.sampling_rate,
            params=default_params,
        )
        return default_waveform

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.actions_taken = 0

        # if random.random() < 0.5:
        #     phi = random.uniform(0.0, np.pi / 4)
        # else:
        #     phi = random.uniform(3 * np.pi / 4, np.pi)

        phi = random.uniform(0.0, np.pi)

        self.state["electrode_radius"] = np.float64(1)
        self.state["theta"] = np.float64(0)
        self.state["phi"] = np.float64(0)
        self.state["electrode_radius"] = np.float64(random.uniform(0.5, 1.5))
        self.state["theta"] = np.float64(random.uniform(0.0, 2 * np.pi))
        self.state["phi"] = np.float64(phi)
        if MULTPLE_NEURON_TYPES:
            self.state["neuron_type"] = int(random.choice(self.neuron_types))
        else:
            self.state["neuron_type"] = int(self.neuron_types[0])

        r = float(self.state["electrode_radius"])
        theta = float(self.state["theta"])
        phi = float(self.state["phi"])
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        if PERTURBATE_ELECTRODE_POSITION:
            sigma = ELECTRODE_POSITION_PERTURBATION_SIGMA
            x += random.gauss(0, sigma)
            y += random.gauss(0, sigma)
            z += random.gauss(0, sigma)

        default_waveform = self.default_stimulation()

        self.elec_field = ICMS(x=x, y=y, z=z, conductivity=0.33)

        types_to_eval = self._active_neuron_types()
        # Push the new field to each active actor and wait for it to land
        # (stimulate depends on the updated xtra params). Then fan out the
        # default waveform so both cells simulate concurrently.
        ray.get(
            [self.actors[nt].set_field.remote(self.elec_field) for nt in types_to_eval]
        )
        wf_ref = ray.put(np.asarray(default_waveform))
        results = ray.get(
            [self.actors[nt].stimulate.remote(wf_ref) for nt in types_to_eval]
        )

        neuron_stimulation_params = []
        for response, _ in results:
            neuron_stimulation_params.extend(self.get_stimulation_params(response))
        self.state["last_stimulation_params"] = np.array(neuron_stimulation_params)

        self.state["last_waveform_params"] = np.zeros(self.waveform.n_params)
        self.state["best_waveform_params"] = np.zeros(self.waveform.n_params)
        self.state["best_reward"] = np.float64(0.0)

        return self.get_obs(), {}

    def close(self):
        actors = getattr(self, "actors", {})
        for a in actors.values():
            try:
                ray.kill(a)
            except Exception:
                pass
        self.actors = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
