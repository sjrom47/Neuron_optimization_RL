from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import gymnasium
from criterions import MinEnergy, SelectivityCriterion
from environment import NeuronEnv

class RecurrentPPOClass:
    def __init__(self, env, waveform, criterion, lr, timesteps):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.lr = lr
        self.timesteps = timesteps
        self.model = RecurrentPPO("MlpLstmPolicy", self.env, learning_rate=self.lr)

    def train(self):
        self.model.learn(total_timesteps=self.timesteps)
        self.model.save("recurrentppo_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward