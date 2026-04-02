from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import gymnasium
from criterions import MinEnergy, SelectivityCriterion
from environment import NeuronEnv

class TD3Class:
    def __init__(self, env, waveform, criterion, lr, sigma, timesteps):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.lr = lr
        self.sigma = sigma
        self.timesteps = timesteps
        n_actions = self.env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.sigma * np.ones(n_actions))
        self.model = TD3("MlpPolicy", self.env, learning_rate=self.lr, action_noise=action_noise)

    def train(self):
        self.model.learn(total_timesteps=self.timesteps)
        self.model.save("td3_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward