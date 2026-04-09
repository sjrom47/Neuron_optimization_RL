import gymnasium
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from criterions import MinEnergy, SelectivityCriterion
from environment import NEURONEnv


# TODO: we should consider refactoring this into an abstract base class for all models
class RecurrentPPOClass:
    def __init__(self, env, waveform, criterion, lr, timesteps):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.lr = lr
        self.timesteps = timesteps
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            self.env,
            learning_rate=self.lr,
            verbose=1,
            n_steps=200,
            batch_size=50,
        )

    def train(self):
        self.model.learn(
            total_timesteps=self.timesteps,
            log_interval=1,
            callback=ProgressBarCallback(),
        )
        self.model.save("weights/recurrentppo_opt")

    def eval(self, eps=1):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
