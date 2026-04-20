import os
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from callbacks import (
    BestResponseCallback,
    DiagnosticsCallback,
    TrainingProgressCallback,
)
from criterions import MinEnergy, SelectivityCriterion
from environment import NEURONEnv


class SACClass:
    def __init__(self, env, waveform, criterion, lr, timesteps, cell_id=36):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.cell_id = cell_id
        self.lr = lr
        self.timesteps = timesteps
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=self.lr,
            buffer_size=200000,
            learning_starts=5000,
            batch_size=256,
            train_freq=1,
            gradient_steps=4,
            gamma=0.0,
            verbose=1,
        )

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("plots", f"{timestamp}_sac_cell{self.cell_id}")
        self.model.learn(
            total_timesteps=self.timesteps,
            log_interval=1,
            callback=CallbackList(
                [
                    ProgressBarCallback(),
                    BestResponseCallback(run_dir=run_dir),
                    TrainingProgressCallback(run_dir=run_dir),
                    DiagnosticsCallback(run_dir=run_dir),
                ]
            ),
        )
        self.model.save(f"weights/sac_{self.waveform}_{self.criterion}_cell{self.cell_id}_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
