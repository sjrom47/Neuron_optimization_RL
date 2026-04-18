import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy

from callbacks import BestResponseCallback, TrainingProgressCallback


class PPOClass:
    def __init__(self, env, waveform, criterion, lr, timesteps):
        self.env = env
        self.waveform = waveform
        self.criterion = criterion
        self.lr = lr
        self.timesteps = timesteps
        self.model = PPO(
            "MlpPolicy",
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
        run_dir = os.path.join("plots", f"{timestamp}_ppo")
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
        self.model.save("weights/ppo_opt")

    def eval(self, eps=10):
        avg_reward = evaluate_policy(self.model, self.env, n_eval_episodes=eps)
        print("Average Reward:", avg_reward)
        return avg_reward
