import os
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ProgressBarCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import ActionNoise

from simulation.callbacks import (
    BestResponseCallback,
    DiagnosticsCallback,
    EpisodeStepRewardCallback,
    TrainingProgressCallback,
    WaveformSnapshotCallback,
)
from simulation.criteria import MinEnergy, SelectivityCriterion
from simulation.environment import NEURONEnv
from simulation.paths import PLOTS_DIR, WEIGHTS_DIR, ensure_dir


class LinearAnnealingNoise(ActionNoise):
    """NormalActionNoise whose sigma decays linearly from sigma_initial to sigma_final."""

    def __init__(self, mean, sigma_initial, sigma_final, total_timesteps):
        super().__init__()
        self._mu = mean
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.total_timesteps = total_timesteps
        self._sigma = sigma_initial * np.ones_like(mean)

    def update(self, current_timestep):
        frac = min(current_timestep / self.total_timesteps, 1.0)
        self._sigma[:] = self.sigma_initial + frac * (
            self.sigma_final - self.sigma_initial
        )

    def __call__(self):
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self):
        return (
            f"LinearAnnealingNoise(sigma={self._sigma[0]:.4f}, "
            f"initial={self.sigma_initial}, final={self.sigma_final})"
        )


class NoiseAnnealCallback(BaseCallback):
    def _on_step(self):
        noise = getattr(self.model, "action_noise", None)
        if isinstance(noise, LinearAnnealingNoise):
            noise.update(self.num_timesteps)
        return True


class PeriodicSaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq=5000):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.save_path)
        return True


class TD3Class:
    def __init__(
        self,
        env,
        waveform,
        criterion,
        lr,
        timesteps,
        cell_id=36,
        sigma_initial=0.2,
        sigma_final=0.15,
    ):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.cell_id = cell_id
        self.lr = lr
        self.timesteps = timesteps
        n_actions = self.env.action_space.shape[0]
        action_noise = LinearAnnealingNoise(
            mean=np.zeros(n_actions),
            sigma_initial=sigma_initial,
            sigma_final=sigma_final,
            total_timesteps=timesteps,
        )
        self.model = TD3(
            "MlpPolicy",
            self.env,
            learning_rate=self.lr,
            action_noise=action_noise,
            buffer_size=50000,
            learning_starts=5000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=8,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_delay=2,
            tau=0.005,
            gamma=0.5,
            verbose=1,
        )

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(ensure_dir(PLOTS_DIR / f"{timestamp}_td3_cell{self.cell_id}"))
        save_path = (
            str(ensure_dir(WEIGHTS_DIR) / f"td3_{self.waveform}_{self.criterion}_cell{self.cell_id}_opt")
        )
        self.model.learn(
            total_timesteps=self.timesteps,
            callback=CallbackList(
                [
                    ProgressBarCallback(),
                    NoiseAnnealCallback(),
                    PeriodicSaveCallback(save_path=save_path, save_freq=5000),
                    BestResponseCallback(run_dir=run_dir),
                    TrainingProgressCallback(run_dir=run_dir),
                    DiagnosticsCallback(run_dir=run_dir),
                    WaveformSnapshotCallback(run_dir=run_dir),
                    EpisodeStepRewardCallback(run_dir=run_dir),
                ]
            ),
            log_interval=1,
        )
        self.model.save(
            str(ensure_dir(WEIGHTS_DIR) / f"td3_{self.waveform}_{self.criterion}_cell{self.cell_id}_opt")
        )

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
