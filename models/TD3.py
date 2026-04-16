import os
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from callbacks import BestResponseCallback
from criterions import MinEnergy, SelectivityCriterion
from environment import NEURONEnv


class TD3Class:
    def __init__(self, env, waveform, criterion, lr, timesteps, sigma=0.1):
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
            learning_starts=2000,
            train_freq=4,
        )

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("plots", f"{timestamp}_td3")
        self.model.learn(
            total_timesteps=self.timesteps,
            callback=CallbackList(
                [ProgressBarCallback(), BestResponseCallback(run_dir=run_dir)]
            ),
            log_interval=1,
        )
        self.model.save("td3_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
