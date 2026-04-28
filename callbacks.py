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
    """Writes one CSV row per completed episode for per-episode diagnostics."""

    def __init__(self, run_dir="plots"):
        super().__init__()
        self.run_dir = run_dir
        self.csv_path = os.path.join(self.run_dir, "debug_metrics.csv")
        self._field_names = [
            "episode",
            "timesteps",
            "episode_return",
            "episode_length",
            "spikes_mean",
            "state_fr_mean",
            "state_peak_vm_mean",
            "state_last_mp_mean",
            "electrode_radius",
            "theta",
            "phi",
            "period_mean",
            "amplitude_mean",
            "duty_cycle_mean",
            "delay_mean",
            "action_abs_mean",
            "action_std",
            "train_actor_loss",
            "train_critic_loss",
        ]
        self._episode_count = 0
        self._env_accumulators = None

    def _init_accumulator(self):
        return {
            "reward": 0.0,
            "length": 0,
            "spikes": [],
            "state_fr": [],
            "state_peak_vm": [],
            "state_last_mp": [],
            "electrode_radius": np.nan,
            "theta": np.nan,
            "phi": np.nan,
            "period": [],
            "amplitude": [],
            "duty_cycle": [],
            "delay": [],
            "action_abs": [],
            "action_vals": [],
        }

    def _on_training_start(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._field_names)

    @staticmethod
    def _safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else np.nan

    @staticmethod
    def _safe_param_value(value):
        try:
            return float(np.mean(np.asarray(value, dtype=float)))
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
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        actions = self.locals.get("actions")
        if actions is not None:
            actions = np.asarray(actions, dtype=float)

        n_envs = len(rewards)
        if self._env_accumulators is None:
            self._env_accumulators = [self._init_accumulator() for _ in range(n_envs)]

        rows = []
        for i in range(n_envs):
            acc = self._env_accumulators[i]
            info = infos[i] if i < len(infos) else {}

            acc["reward"] += float(rewards[i])
            acc["length"] += 1

            if info.get("spikes") is not None:
                acc["spikes"].append(float(info["spikes"]))
            if info.get("state_fr") is not None:
                acc["state_fr"].append(float(info["state_fr"]))
            if info.get("state_peak_vm") is not None:
                acc["state_peak_vm"].append(float(info["state_peak_vm"]))
            if info.get("state_last_mp") is not None:
                acc["state_last_mp"].append(float(info["state_last_mp"]))
            if info.get("electrode_radius") is not None:
                acc["electrode_radius"] = float(info["electrode_radius"])
            if info.get("theta") is not None:
                acc["theta"] = float(info["theta"])
            if info.get("phi") is not None:
                acc["phi"] = float(info["phi"])

            params = info.get("params")
            if isinstance(params, dict):
                for key in ("period", "amplitude", "duty_cycle", "delay"):
                    if key in params:
                        acc[key].append(self._safe_param_value(params[key]))

            if actions is not None and i < len(actions):
                acc["action_abs"].append(float(np.mean(np.abs(actions[i]))))
                acc["action_vals"].extend(actions[i].reshape(-1).tolist())

            if dones[i]:
                self._episode_count += 1
                rows.append({
                    "episode": self._episode_count,
                    "timesteps": int(self.num_timesteps),
                    "episode_return": acc["reward"],
                    "episode_length": acc["length"],
                    "spikes_mean": self._safe_mean(acc["spikes"]),
                    "state_fr_mean": self._safe_mean(acc["state_fr"]),
                    "state_peak_vm_mean": self._safe_mean(acc["state_peak_vm"]),
                    "state_last_mp_mean": self._safe_mean(acc["state_last_mp"]),
                    "electrode_radius": acc["electrode_radius"],
                    "theta": acc["theta"],
                    "phi": acc["phi"],
                    "period_mean": self._safe_mean(acc["period"]),
                    "amplitude_mean": self._safe_mean(acc["amplitude"]),
                    "duty_cycle_mean": self._safe_mean(acc["duty_cycle"]),
                    "delay_mean": self._safe_mean(acc["delay"]),
                    "action_abs_mean": self._safe_mean(acc["action_abs"]),
                    "action_std": float(np.std(acc["action_vals"])) if acc["action_vals"] else np.nan,
                    "train_actor_loss": self._logger_value("train/actor_loss"),
                    "train_critic_loss": self._logger_value("train/critic_loss"),
                })
                self._env_accumulators[i] = self._init_accumulator()

        if rows:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._field_names)
                writer.writerows(rows)

        return True


class WaveformSnapshotCallback(BaseCallback):
    """Saves a grid of recent waveforms every snap_freq episodes so you can
    see what the policy is actually generating throughout training."""

    def __init__(self, run_dir="plots", snap_freq=20, n_waveforms=8):
        super().__init__()
        self.run_dir = run_dir
        self.snap_freq = snap_freq
        self.n_waveforms = n_waveforms
        self._episode_count = 0
        self._buffer = []  # list of (waveform, reward, params, sampling_rate)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)

        for i, info in enumerate(infos):
            waveform = info.get("waveform")
            if waveform is None:
                continue
            self._buffer.append((
                np.array(waveform, copy=True),
                float(rewards[i]),
                dict(info.get("params", {})),
                float(info.get("sampling_rate", 1.0)),
                float(info.get("max_amplitude", 500.0)),
            ))
            if len(self._buffer) > self.n_waveforms:
                self._buffer.pop(0)

            if dones[i]:
                self._episode_count += 1
                if self._episode_count % self.snap_freq == 0:
                    self._save_snapshot()

        return True

    def _on_training_end(self):
        self._save_snapshot()

    def _save_snapshot(self):
        if not self._buffer:
            return

        snap_dir = os.path.join(self.run_dir, "waveform_snapshots")
        os.makedirs(snap_dir, exist_ok=True)

        n = len(self._buffer)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(f"Waveform Snapshots — Episode {self._episode_count}", fontsize=12)

        for idx, (waveform, reward, params, sr, max_amp) in enumerate(self._buffer):
            row, col = divmod(idx, ncols)
            ax = axs[row][col]
            t = np.arange(len(waveform)) / sr * 1000.0
            ax.plot(t, waveform, linewidth=0.8)
            ax.set_ylim(-1.1 * max_amp, 1.1 * max_amp)
            ax.set_xlabel("Time (ms)", fontsize=7)
            ax.set_ylabel("Amp (µA)", fontsize=7)
            param_str = "  ".join(f"{k}={v:.2f}" for k, v in list(params.items())[:3])
            ax.set_title(f"r={reward:.2f}\n{param_str}", fontsize=7)
            ax.tick_params(labelsize=6)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axs[row][col].axis("off")

        plt.tight_layout()
        fname = os.path.join(snap_dir, f"ep_{self._episode_count:05d}.png")
        plt.savefig(fname, dpi=100)
        plt.close()


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
        for key in ("all_responses", "all_times"):
            if key in copied and copied[key] is not None:
                copied[key] = [np.array(r, copy=True) for r in copied[key]]
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

        all_responses = info.get("all_responses")
        all_times = info.get("all_times")
        neuron_types = info.get("neuron_types")
        multi = (
            all_responses is not None
            and neuron_types is not None
            and len(all_responses) > 1
        )

        os.makedirs(self.run_dir, exist_ok=True)
        t_waveform = np.arange(len(waveform)) / sampling_rate * 1000  # ms
        param_str = " ".join(f"{k}={self._format_param(v)}" for k, v in params.items())

        n_response_rows = len(all_responses) if multi else 1
        fig, axs = plt.subplots(1 + n_response_rows, 1, figsize=(12, 3 * (1 + n_response_rows)), sharex=False)
        fig.suptitle(f"Params: {param_str}\nReward: {reward:.3f}")

        axs[0].plot(t_waveform, waveform)
        axs[0].set_title("Stimulation Waveform")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_ylim(-1.1 * max_amplitude, 1.1 * max_amplitude)

        if multi:
            for i, (resp, t, nt) in enumerate(zip(all_responses, all_times, neuron_types)):
                axs[1 + i].plot(np.asarray(t, dtype=float), np.asarray(resp, dtype=float))
                axs[1 + i].set_title(f"Neuron {nt} Response")
                axs[1 + i].set_xlabel("Time (ms)")
                axs[1 + i].set_ylabel("Voltage (mV)")
                axs[1 + i].set_ylim(-100, 40)
        else:
            axs[1].plot(time_response, response)
            axs[1].set_title("Neuron Response")
            axs[1].set_xlabel("Time (ms)")
            axs[1].set_ylabel("Voltage (mV)")
            axs[1].set_ylim(-100, 40)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{plot_name}.png"))
        plt.close()
