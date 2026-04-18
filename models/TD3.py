import os
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import TD3
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


class TD3Class:
    def __init__(self, env, waveform, criterion, lr, timesteps, sigma=0.15):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.lr = lr
        self.sigma = sigma
        self.timesteps = timesteps
        n_actions = self.env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=self.sigma * np.ones(n_actions)
        )
        self.model = TD3(
            "MlpPolicy",
            self.env,
            learning_rate=self.lr,
            action_noise=action_noise,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=8,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_delay=2,
            tau=0.005,
            gamma=0.0,
            verbose=1,
        )

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("plots", f"{timestamp}_td3")
        self.model.learn(
            total_timesteps=self.timesteps,
            callback=CallbackList(
                [
                    ProgressBarCallback(),
                    BestResponseCallback(run_dir=run_dir),
                    TrainingProgressCallback(run_dir=run_dir),
                    DiagnosticsCallback(run_dir=run_dir),
                ]
            ),
            log_interval=1,
        )
        self.model.save("td3_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
