import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BestResponseCallback(BaseCallback):
    """Tracks the global best reward across all parallel workers and saves plots."""

    def __init__(self, run_dir="plots"):
        super().__init__()
        self.best_reward = -np.inf
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            reward = info.get("reward")
            if reward is None:
                continue

            if reward > self.best_reward:
                self.best_reward = reward
                self._plot(info, plot_name="best_response")

            if done and info.get("terminated"):
                self._plot(info, plot_name="terminated")

        return True

    def _plot(self, info, plot_name):
        waveform = info["waveform"]
        response = info["response"]
        time_response = info["time_response"]
        params = info["params"]
        reward = info["reward"]
        max_amplitude = info["max_amplitude"]
        sampling_rate = info["sampling_rate"]

        os.makedirs(self.run_dir, exist_ok=True)
        t_waveform = np.arange(len(waveform)) / sampling_rate * 1000  # ms

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        fig.suptitle(
            f"Params: {' '.join([f'{k}={v:.2f}' for k, v in params.items()])}\n"
            f"Reward: {reward:.3f}"
        )
        axs[0].plot(t_waveform, waveform)
        axs[0].set_title("Stimulation Waveform")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_ylim(-1.1 * max_amplitude, 1.1 * max_amplitude)

        axs[1].plot(time_response, response)
        axs[1].set_title("Neuron Response")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Voltage (mV)")
        axs[1].set_ylim(-100, 40)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{plot_name}.png"))
        plt.close()
