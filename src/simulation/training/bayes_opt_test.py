import os

import matplotlib.pyplot as plt
import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from tqdm import tqdm

from simulation.environment import NEURONEnv
from simulation.paths import PLOTS_DIR, ensure_dir

N_RANDOM = 20
N_BO = 500


def normalize_params(params, waveform):
    """Convert actual parameter values to normalized [-1, 1] action array."""
    return np.array(
        [
            waveform.normalize_param(params[key], key)
            for key in waveform.param_bounds.keys()
        ]
    )


def run_trial(env, params):
    action = normalize_params(params, env.waveform)
    obs, reward, terminated, truncated, info = env.step(action)
    return reward


def plot_results(history):
    plot_dir = ensure_dir(PLOTS_DIR / "B")
    iterations = range(1, len(history) + 1)
    rewards = [r for r, _ in history]
    params_list = [p for _, p in history]

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, rewards, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Reward over Iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / "reward.png")
    plt.close()

    param_names = list(params_list[0].keys())
    for name in param_names:
        values = [p[name] for p in params_list]
        plt.figure(figsize=(8, 4))
        plt.plot(iterations, values, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel(name)
        plt.title(f"{name} over Iterations")
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir / f"{name}.png")
        plt.close()


if __name__ == "__main__":
    ensure_dir(PLOTS_DIR / "B")

    # max_actions set high so env never terminates during the BO run
    env = NEURONEnv(
        waveform_type="square",
        criterion_type="min_energy",
        max_actions=N_RANDOM + N_BO + 10,
    )
    env.reset()

    param_bounds = env.waveform.param_bounds

    client = Client()
    client.configure_experiment(
        name="TI_optimization",
        parameters=[
            RangeParameterConfig(
                name=name,
                bounds=(float(lo), float(hi)),
                parameter_type="float",
            )
            for name, (lo, hi) in param_bounds.items()
        ],
    )
    client.configure_optimization(objective="-1*reward")

    history = []

    best_reward = -np.inf

    # Random exploration phase
    print(f"Running {N_RANDOM} random trials...")
    for i in tqdm(range(N_RANDOM)):
        random_params = {
            key: float(np.random.uniform(lo, hi))
            for key, (lo, hi) in param_bounds.items()
        }
        reward = run_trial(env, random_params)
        trial_index = client.attach_trial(parameters=random_params)
        client.complete_trial(trial_index=trial_index, raw_data={"reward": reward})
        history.append((reward, random_params))
        if reward > best_reward:
            best_reward = reward
            print(f"New best reward: {best_reward:.4f} at iteration {i+1}")

    # Bayesian optimization phase
    print(f"\nRunning {N_BO} Bayesian optimization trials...")
    for i in tqdm(range(N_BO)):
        for trial_index, parameters in client.get_next_trials(max_trials=1).items():
            reward = run_trial(env, parameters)
            client.complete_trial(
                trial_index=trial_index,
                raw_data={"reward": reward},
            )
            history.append((reward, parameters))
            if reward > best_reward:
                best_reward = reward
                print(
                    f"New best reward: {best_reward:.4f} at iteration {N_RANDOM + i + 1}"
                )
            print(
                f"Iteration {N_RANDOM + i + 1}: Reward = {reward:.4f}, Best Reward = {best_reward:.4f}"
            )

    best_parameterization = client.get_best_parameterization()
    print("\nBest Parameters:", best_parameterization)

    plot_results(history)
