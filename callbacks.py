import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """Records per-episode returns across vectorized envs and saves a progress plot."""

    def __init__(self, run_dir="plots", plot_freq=10, step_plot_freq=100):
        super().__init__()
        self.run_dir = run_dir
        self.plot_freq = plot_freq
        self.step_plot_freq = step_plot_freq
        self._episode_returns = []
        self._episode_lengths = []
        self._episode_indices = []
        self._best_returns = []
        self._best_return = -np.inf
        self._episode_count = 0
        self._env_returns = None
        self._env_lengths = None
        self._step_means = []
        self._step_mins = []
        self._step_maxs = []
        self._step_timesteps = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=float).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)

        self._step_timesteps.append(self.num_timesteps)
        self._step_means.append(float(np.mean(rewards)))
        self._step_mins.append(float(np.min(rewards)))
        self._step_maxs.append(float(np.max(rewards)))

        if len(self._step_timesteps) % self.step_plot_freq == 0:
            self._save_step_plot()

        if self._env_returns is None or len(self._env_returns) != len(rewards):
            self._env_returns = np.zeros_like(rewards, dtype=float)
            self._env_lengths = np.zeros_like(rewards, dtype=int)

        self._env_returns += rewards
        self._env_lengths += 1

        for env_idx, done in enumerate(dones):
            if not done:
                continue

            episode_return = float(self._env_returns[env_idx])
            episode_length = int(self._env_lengths[env_idx])

            self._episode_count += 1
            self._episode_returns.append(episode_return)
            self._episode_lengths.append(episode_length)
            self._episode_indices.append(self._episode_count)

            if episode_return > self._best_return:
                self._best_return = episode_return
            self._best_returns.append(self._best_return)

            self._env_returns[env_idx] = 0.0
            self._env_lengths[env_idx] = 0

            if self._episode_count % self.plot_freq == 0:
                self._save_plot()

        return True

    def _on_training_end(self):
        self._save_plot()
        self._save_step_plot()

    def _save_plot(self):
        if not self._episode_returns:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        returns = np.array(self._episode_returns)
        episode_idx = np.array(self._episode_indices)

        window = min(20, len(returns))
        rolling_mean = np.convolve(returns, np.ones(window) / window, mode="valid")
        rolling_ep = episode_idx[window - 1 :]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(
            episode_idx,
            returns,
            alpha=0.3,
            s=10,
            color="steelblue",
            label="Episode return",
        )
        ax.plot(
            rolling_ep,
            rolling_mean,
            color="orange",
            linewidth=2,
            label=f"Rolling mean (n={window})",
        )
        ax.plot(
            episode_idx,
            self._best_returns,
            color="green",
            linewidth=1.5,
            linestyle="--",
            label="Best episode return",
        )
        ax.set_xlabel("Completed Episodes")
        ax.set_ylabel("Return")
        ax.set_title("Training Progress")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "training_progress.png"))
        plt.close()

    def _save_step_plot(self):
        if not self._step_timesteps:
            return

        os.makedirs(self.run_dir, exist_ok=True)
        timesteps = np.array(self._step_timesteps)
        means = np.array(self._step_means)
        mins = np.array(self._step_mins)
        maxs = np.array(self._step_maxs)

        # Keep rendering fast for long runs while preserving trend visibility.
        max_points = 4000
        if len(timesteps) > max_points:
            idx = np.linspace(0, len(timesteps) - 1, max_points, dtype=int)
            timesteps = timesteps[idx]
            means = means[idx]
            mins = mins[idx]
            maxs = maxs[idx]

        window = min(50, len(means))
        rolling_mean = np.convolve(means, np.ones(window) / window, mode="valid")
        rolling_ts = timesteps[window - 1 :]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            timesteps,
            means,
            color="steelblue",
            linewidth=1,
            alpha=0.5,
            label="Mean step reward",
        )
        ax.fill_between(
            timesteps,
            mins,
            maxs,
            color="lightsteelblue",
            alpha=0.3,
            label="Min/Max across envs",
        )
        ax.plot(
            rolling_ts,
            rolling_mean,
            color="orange",
            linewidth=2,
            label=f"Rolling mean (n={window})",
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Step-Level Reward Progress")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "training_step_progress.png"))
        plt.close()


class DiagnosticsCallback(BaseCallback):
    """Writes periodic training diagnostics to CSV for debugging instability."""

    def __init__(self, run_dir="plots", log_freq=50):
        super().__init__()
        self.run_dir = run_dir
        self.log_freq = log_freq
        self.csv_path = os.path.join(self.run_dir, "debug_metrics.csv")
        self._field_names = [
            "timesteps",
            "callback_step",
            "reward_mean",
            "reward_min",
            "reward_max",
            "done_rate",
            "action_abs_mean",
            "action_std",
            "spikes_mean",
            "state_fr_mean",
            "state_peak_vm_mean",
            "state_last_mp_mean",
            "period_mean",
            "amplitude_mean",
            "duty_cycle_mean",
            "delay_mean",
            "train_actor_loss",
            "train_critic_loss",
            "train_ent_coef",
            "train_ent_coef_loss",
        ]

    def _on_training_start(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._field_names)

    @staticmethod
    def _safe_mean(values):
        if len(values) == 0:
            return np.nan
        return float(np.mean(values))

    @staticmethod
    def _safe_param_value(value):
        try:
            arr = np.asarray(value, dtype=float)
            return float(np.mean(arr))
        except Exception:
            return np.nan

    def _logger_value(self, key):
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return np.nan
        values = getattr(logger, "name_to_value", None)
        if not isinstance(values, dict):
            return np.nan
        value = values.get(key)
        if value is None:
            return np.nan
        try:
            return float(value)
        except Exception:
            return np.nan

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        actions = self.locals.get("actions")

        action_abs_mean = np.nan
        action_std = np.nan
        if actions is not None:
            actions = np.asarray(actions, dtype=float)
            action_abs_mean = float(np.mean(np.abs(actions)))
            action_std = float(np.std(actions))

        spikes = []
        state_frs = []
        state_peak_vms = []
        state_last_mps = []
        periods = []
        amplitudes = []
        duty_cycles = []
        delays = []

        for info in infos:
            if info.get("spikes") is not None:
                spikes.append(float(info["spikes"]))
            if info.get("state_fr") is not None:
                state_frs.append(float(info["state_fr"]))
            if info.get("state_peak_vm") is not None:
                state_peak_vms.append(float(info["state_peak_vm"]))
            if info.get("state_last_mp") is not None:
                state_last_mps.append(float(info["state_last_mp"]))

            params = info.get("params")
            if isinstance(params, dict):
                if "period" in params:
                    periods.append(self._safe_param_value(params["period"]))
                if "amplitude" in params:
                    amplitudes.append(self._safe_param_value(params["amplitude"]))
                if "duty_cycle" in params:
                    duty_cycles.append(self._safe_param_value(params["duty_cycle"]))
                if "delay" in params:
                    delays.append(self._safe_param_value(params["delay"]))

        row = {
            "timesteps": int(self.num_timesteps),
            "callback_step": int(self.n_calls),
            "reward_mean": float(np.mean(rewards)) if rewards.size > 0 else np.nan,
            "reward_min": float(np.min(rewards)) if rewards.size > 0 else np.nan,
            "reward_max": float(np.max(rewards)) if rewards.size > 0 else np.nan,
            "done_rate": float(np.mean(dones)) if dones.size > 0 else np.nan,
            "action_abs_mean": action_abs_mean,
            "action_std": action_std,
            "spikes_mean": self._safe_mean(spikes),
            "state_fr_mean": self._safe_mean(state_frs),
            "state_peak_vm_mean": self._safe_mean(state_peak_vms),
            "state_last_mp_mean": self._safe_mean(state_last_mps),
            "period_mean": self._safe_mean(periods),
            "amplitude_mean": self._safe_mean(amplitudes),
            "duty_cycle_mean": self._safe_mean(duty_cycles),
            "delay_mean": self._safe_mean(delays),
            "train_actor_loss": self._logger_value("train/actor_loss"),
            "train_critic_loss": self._logger_value("train/critic_loss"),
            "train_ent_coef": self._logger_value("train/ent_coef"),
            "train_ent_coef_loss": self._logger_value("train/ent_coef_loss"),
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._field_names)
            writer.writerow(row)

        return True


class BestResponseCallback(BaseCallback):
    """Tracks the global best single-step response across parallel workers."""

    def __init__(self, run_dir="plots"):
        super().__init__()
        self.best_reward = -np.inf
        self.run_dir = run_dir

    @staticmethod
    def _format_param(value):
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return str(value)

        if arr.ndim == 0:
            return f"{float(arr):.2f}"

        flat = arr.reshape(-1)
        shown = ", ".join(f"{float(v):.2f}" for v in flat[:4])
        if flat.size > 4:
            shown += ", ..."
        return f"[{shown}]"

    @staticmethod
    def _copy_info(info):
        if not isinstance(info, dict):
            return {}

        copied = dict(info)
        for key in ("waveform", "response", "time_response"):
            if key in copied and copied[key] is not None:
                copied[key] = np.array(copied[key], copy=True)
        if isinstance(copied.get("params"), dict):
            copied["params"] = dict(copied["params"])
        return copied

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        infos = self.locals.get("infos", [])

        if rewards.size == 0:
            return True

        for env_idx, reward in enumerate(rewards):
            info = infos[env_idx] if env_idx < len(infos) else {}
            step_reward = float(reward)

            if step_reward > self.best_reward:
                self.best_reward = step_reward
                step_info = self._copy_info(info)
                step_info["reward"] = step_reward
                self._plot(step_info, plot_name="best_response")

        return True

    def _plot(self, info, plot_name):
        waveform = np.asarray(info.get("waveform", []), dtype=float)
        response = np.asarray(info.get("response", []), dtype=float)
        time_response = np.asarray(info.get("time_response", []), dtype=float)
        params = info.get("params", {})
        reward = float(info.get("reward", np.nan))
        max_amplitude = float(
            info.get(
                "max_amplitude", np.max(np.abs(waveform)) if waveform.size else 1.0
            )
        )
        sampling_rate = float(info.get("sampling_rate", 1.0))

        if waveform.size == 0 or response.size == 0 or time_response.size == 0:
            return
        if sampling_rate <= 0:
            sampling_rate = 1.0

        os.makedirs(self.run_dir, exist_ok=True)
        t_waveform = np.arange(len(waveform)) / sampling_rate * 1000  # ms

        param_str = " ".join(f"{k}={self._format_param(v)}" for k, v in params.items())

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        fig.suptitle(f"Params: {param_str}\n" f"Reward: {reward:.3f}")
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
