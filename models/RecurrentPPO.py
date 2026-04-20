import os
from datetime import datetime

import gymnasium
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from callbacks import BestResponseCallback, TrainingProgressCallback
from criterions import MinEnergy, SelectivityCriterion
from environment import NEURONEnv


# TODO: we should consider refactoring this into an abstract base class for all models
class RecurrentPPOClass:
    def __init__(self, env, waveform, criterion, lr, timesteps, cell_id=36):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.cell_id = cell_id
        self.lr = lr
        self.timesteps = timesteps
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            self.env,
            learning_rate=self.lr,
            verbose=1,
            n_steps=256,
            batch_size=128,
            gamma=0.0,
            gae_lambda=0.0,
        )

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("plots", f"{timestamp}_recurrentppo_cell{self.cell_id}")
        self.model.learn(
            total_timesteps=self.timesteps,
            log_interval=1,
            callback=CallbackList(
                [
                    ProgressBarCallback(),
                    BestResponseCallback(run_dir=run_dir),
                    TrainingProgressCallback(run_dir=run_dir),
                ]
            ),
        )
        self.model.save(f"weights/recurrentppo_{self.waveform}_{self.criterion}_cell{self.cell_id}_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
