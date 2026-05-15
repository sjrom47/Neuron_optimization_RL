"""Compare a trained RL policy against an Ax Bayesian-optimization baseline."""

import argparse
import os
import random
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from simulation.environment import NEURONEnv
from simulation.paths import PLOTS_DIR, WEIGHTS_DIR

MODEL_CLASSES = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
}


def default_model_path_candidates(model_type, waveform_type, criterion_type, cell_id=36):
    stem = str(WEIGHTS_DIR / f"{model_type}_{waveform_type}_{criterion_type}_cell{cell_id}_opt")
    return [f"{stem}.zip", stem]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a trained RL model against Ax BayesOpt using best single-step "
            "reward per episode."
        )
    )
    parser.add_argument(
        "--waveform_type",
        type=str,
        default="fourier",
        choices=["fourier", "legendre3", "square", "two_sines", "charge_balanced"],
        help="Waveform type for evaluation environment.",
    )
    parser.add_argument(
        "--criterion_type",
        type=str,
        default="min_energy",
        choices=["min_energy", "selectivity"],
        help="Criterion type for evaluation environment.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ppo",
        choices=["ppo", "recurrentppo", "sac", "td3", "tqc"],
        help="RL model type to load.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.zip). If omitted, defaults are tried.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for each method.",
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=10,
        help="Max actions (steps) per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Episode k uses seed + k for both methods.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PLOTS_DIR / "compare_vs_bayesopt_baseline"),
        help="Directory where plots and comparison outputs are saved.",
    )
    parser.add_argument(
        "--cell_id",
        type=int,
        default=36,
        help="Primary neuron cell ID used to locate the trained model weights.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.max_actions < 1:
        raise ValueError("--max_actions must be >= 1")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def normalize_params(params, waveform):
    """Convert unnormalized parameter values to normalized [-1, 1] action array."""
    learnable_keys = list(waveform.param_bounds.keys())[: waveform.n_params]
    return np.array(
        [waveform.normalize_param(params[key], key) for key in learnable_keys],
        dtype=float,
    )


def resolve_model_path(model_type, waveform_type, criterion_type, model_path, cell_id=36):
    def _existing_checkpoint(candidates):
        for candidate in candidates:
            expanded = os.path.abspath(os.path.expanduser(candidate))
            if os.path.isfile(expanded):
                return expanded
        return None

    if model_path is not None:
        model_path = os.path.expanduser(model_path)
        candidates = [model_path]
        if not model_path.endswith(".zip"):
            candidates.append(f"{model_path}.zip")
        found = _existing_checkpoint(candidates)
        if found is None:
            raise FileNotFoundError(
                "Could not find checkpoint at the provided --model_path. "
                f"Tried: {candidates}"
            )
        return found

    candidates = default_model_path_candidates(model_type, waveform_type, criterion_type, cell_id)
    found = _existing_checkpoint(candidates)
    if found is None:
        raise FileNotFoundError(
            f"Could not find a default checkpoint for {model_type}/{waveform_type}/{criterion_type}. "
            f"Tried: {candidates}. Provide --model_path explicitly."
        )
    return found


def make_ax_client(param_bounds, episode_idx):
    client = Client()
    client.configure_experiment(
        name=f"TI_optimization_ep_{episode_idx}",
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
    return client


def copy_step_info(info, reward):
    copied = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        elif isinstance(value, dict):
            copied[key] = dict(value)
        else:
            copied[key] = value
    copied["reward"] = float(reward)
    return copied


def save_episode_waveform_comparison_plots(args, run_dir, rl_results, bo_results):
    rl_infos = rl_results["episode_best_infos"]
    bo_infos = bo_results["episode_best_infos"]

    if len(rl_infos) != len(bo_infos):
        raise RuntimeError(
            "Episode info count mismatch between RL and BayesOpt results. "
            f"RL={len(rl_infos)}, BayesOpt={len(bo_infos)}"
        )

    plot_dir = os.path.join(run_dir, "episode_waveforms")
    os.makedirs(plot_dir, exist_ok=True)

    plot_paths = []
    for ep_idx, (rl_info, bo_info) in enumerate(zip(rl_infos, bo_infos), start=1):
        rl_waveform = np.asarray(rl_info.get("waveform", []), dtype=float)
        bo_waveform = np.asarray(bo_info.get("waveform", []), dtype=float)

        if rl_waveform.size == 0 or bo_waveform.size == 0:
            raise ValueError(
                f"Missing waveform data for episode {ep_idx}. "
                "Cannot save episode waveform comparison plot."
            )

        rl_reward = float(rl_info.get("reward", np.nan))
        bo_reward = float(bo_info.get("reward", np.nan))

        rl_sampling_rate = float(rl_info.get("sampling_rate", 1.0))
        bo_sampling_rate = float(bo_info.get("sampling_rate", 1.0))
        if rl_sampling_rate <= 0:
            rl_sampling_rate = 1.0
        if bo_sampling_rate <= 0:
            bo_sampling_rate = 1.0

        t_rl = np.arange(len(rl_waveform)) / rl_sampling_rate * 1000.0
        t_bo = np.arange(len(bo_waveform)) / bo_sampling_rate * 1000.0

        rl_max_amplitude = float(
            rl_info.get(
                "max_amplitude",
                np.max(np.abs(rl_waveform)) if rl_waveform.size else 1.0,
            )
        )
        bo_max_amplitude = float(
            bo_info.get(
                "max_amplitude",
                np.max(np.abs(bo_waveform)) if bo_waveform.size else 1.0,
            )
        )

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        fig.suptitle(f"Episode {ep_idx} Best Generated Waveforms")

        axs[0].plot(t_rl, rl_waveform)
        axs[0].set_title(f"RL ({args.model_type}) | Reward: {rl_reward:.4f}")
        axs[0].set_xlabel("Time (ms)")
        axs[0].set_ylabel("Amplitude")
        if rl_max_amplitude > 0:
            axs[0].set_ylim(-1.1 * rl_max_amplitude, 1.1 * rl_max_amplitude)

        axs[1].plot(t_bo, bo_waveform)
        axs[1].set_title(f"BayesOpt | Reward: {bo_reward:.4f}")
        axs[1].set_xlabel("Time (ms)")
        axs[1].set_ylabel("Amplitude")
        if bo_max_amplitude > 0:
            axs[1].set_ylim(-1.1 * bo_max_amplitude, 1.1 * bo_max_amplitude)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"episode_{ep_idx:02d}_waveforms.png")
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)

    return plot_paths


def save_comparison_summary_plots(args, run_dir, rl_results, bo_results):
    """Save aggregate RL vs BayesOpt figures alongside per-episode waveform plots."""
    rl_ep = np.asarray(rl_results["episode_best_rewards"], dtype=float)
    bo_ep = np.asarray(bo_results["episode_best_rewards"], dtype=float)
    if rl_ep.shape != bo_ep.shape:
        raise RuntimeError("Episode reward shape mismatch for summary plots.")
    n = int(rl_ep.shape[0])
    episodes = np.arange(1, n + 1, dtype=float)
    paths = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.35
    ax.bar(episodes - width / 2, rl_ep, width, label=f"RL ({args.model_type})")
    ax.bar(episodes + width / 2, bo_ep, width, label="BayesOpt")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Best single-step reward")
    ax.set_title("Per-episode best reward (RL vs BayesOpt)")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(run_dir, "summary_episode_rewards.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(10, 4))
    delta = rl_ep - bo_ep
    colors = [("C0" if d >= 0 else "C3") for d in delta]
    ax.bar(episodes, delta, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("RL - BayesOpt")
    ax.set_title("Per-episode reward advantage")
    fig.tight_layout()
    p2 = os.path.join(run_dir, "summary_reward_delta.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    return paths


def project_vec_obs_for_checkpoint(obs, model, inner_env):
    """Map vector-env observations to the shape saved in an older checkpoint.

    The environment observation grew by ``n_params + 1`` (best waveform
    parameters plus best reward). Policies trained before that change expect
    the shorter vector (metadata, last waveform params, last stim params only).
    """
    expected = int(model.observation_space.shape[0])
    got = int(obs.shape[-1])
    if got == expected:
        return obs
    full = int(inner_env.observation_space.shape[0])
    extra = int(inner_env.waveform.n_params) + 1
    if full == got and expected == full - extra:
        return obs[..., :expected]
    raise ValueError(
        "Observation size from the environment does not match this checkpoint. "
        f"env_obs_dim={got}, checkpoint expects {expected}, "
        f"current NEURONEnv declares {full}. "
        "Use a checkpoint trained with the same observation layout, or retrain."
    )


def run_rl_episode(env, model, is_recurrent, inner_env):
    # env is a DummyVecEnv(1): reset returns (1, obs_dim), step returns
    # (obs, rewards, dones, infos) with auto-reset on done.
    obs = project_vec_obs_for_checkpoint(env.reset(), model, inner_env)

    best_reward = -np.inf
    best_info = None

    lstm_states = None
    episode_starts = np.array([True], dtype=bool)
    done = np.array([False])

    while not done[0]:
        obs_in = project_vec_obs_for_checkpoint(obs, model, inner_env)
        if is_recurrent:
            action, lstm_states = model.predict(
                obs_in,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
        else:
            action, _ = model.predict(obs_in, deterministic=True)

        obs, rewards, dones, infos = env.step(action)
        done = dones
        reward = float(rewards[0])
        info = infos[0]

        if is_recurrent:
            episode_starts = done.copy()

        if reward > best_reward:
            best_reward = reward
            best_info = copy_step_info(info, reward)

    return best_reward, best_info


def run_bayesopt_episode(env, episode_idx):
    learnable_bounds = dict(list(env.waveform.param_bounds.items())[: env.waveform.n_params])
    client = make_ax_client(learnable_bounds, episode_idx=episode_idx)

    obs, _ = env.reset()
    _ = obs
    terminated = False
    truncated = False

    best_reward = -np.inf
    best_info = None

    while not (terminated or truncated):
        next_trials = client.get_next_trials(max_trials=1)
        if not next_trials:
            raise RuntimeError("Ax returned no next trial during BayesOpt episode.")

        for trial_index, parameters in next_trials.items():
            action = normalize_params(parameters, env.waveform)
            obs, reward, terminated, truncated, info = env.step(action)
            _ = obs
            reward = float(reward)
            client.complete_trial(trial_index=trial_index, raw_data={"reward": reward})

            if reward > best_reward:
                best_reward = reward
                best_info = copy_step_info(info, reward)

            if terminated or truncated:
                break

    return best_reward, best_info


def evaluate_rl(args, seeds, run_dir):
    env = DummyVecEnv(
        [
            lambda: NEURONEnv(
                waveform_type=args.waveform_type,
                criterion_type=args.criterion_type,
                max_actions=args.max_actions,
            )
        ]
    )

    model_path = resolve_model_path(args.model_type, args.waveform_type, args.criterion_type, args.model_path, args.cell_id)
    model_class = MODEL_CLASSES[args.model_type]
    # Load weights only — no env arg needed for inference, and avoids n_envs
    # mismatch warnings (trained on SubprocVecEnv(12), eval uses DummyVecEnv(1)).
    model = model_class.load(model_path, device="auto")
    is_recurrent = args.model_type == "recurrentppo"
    inner_env = env.envs[0]
    try:
        project_vec_obs_for_checkpoint(
            np.zeros((1, inner_env.observation_space.shape[0]), dtype=np.float32),
            model,
            inner_env,
        )
    except ValueError as exc:
        env.close()
        raise ValueError(
            "Loaded policy is incompatible with the evaluation environment "
            "observation space. See message below."
        ) from exc

    episode_best_rewards = []
    episode_best_infos = []
    global_best_reward = -np.inf
    global_best_info = None

    for ep_idx, seed in enumerate(seeds):
        seed_everything(seed)
        ep_best_reward, ep_best_info = run_rl_episode(
            env, model, is_recurrent, inner_env
        )
        if ep_best_info is None:
            raise RuntimeError(
                f"RL episode {ep_idx + 1} produced no valid best-step info."
            )
        episode_best_rewards.append(ep_best_reward)
        episode_best_infos.append(ep_best_info)

        if ep_best_reward > global_best_reward:
            global_best_reward = ep_best_reward
            global_best_info = ep_best_info

        print(
            f"[RL:{args.model_type}] Episode {ep_idx + 1}/{args.episodes} "
            f"best step reward = {ep_best_reward:.4f}"
        )

    if global_best_info is None:
        raise RuntimeError("RL evaluation produced no valid best-step info.")

    env.close()

    return {
        "episode_best_rewards": np.asarray(episode_best_rewards, dtype=float),
        "episode_best_infos": episode_best_infos,
        "global_best_reward": float(global_best_reward),
        "global_best_info": global_best_info,
        "model_path": model_path,
    }


def evaluate_bayesopt(args, seeds, run_dir):
    env = NEURONEnv(
        waveform_type=args.waveform_type,
        criterion_type=args.criterion_type,
        max_actions=args.max_actions,
    )

    episode_best_rewards = []
    episode_best_infos = []
    global_best_reward = -np.inf
    global_best_info = None

    for ep_idx, seed in enumerate(seeds):
        seed_everything(seed)
        ep_best_reward, ep_best_info = run_bayesopt_episode(env, episode_idx=ep_idx)
        if ep_best_info is None:
            raise RuntimeError(
                f"BayesOpt episode {ep_idx + 1} produced no valid best-step info."
            )
        episode_best_rewards.append(ep_best_reward)
        episode_best_infos.append(ep_best_info)

        if ep_best_reward > global_best_reward:
            global_best_reward = ep_best_reward
            global_best_info = ep_best_info

        print(
            f"[BayesOpt] Episode {ep_idx + 1}/{args.episodes} "
            f"best step reward = {ep_best_reward:.4f}"
        )

    if global_best_info is None:
        raise RuntimeError("BayesOpt evaluation produced no valid best-step info.")

    env.close()

    return {
        "episode_best_rewards": np.asarray(episode_best_rewards, dtype=float),
        "episode_best_infos": episode_best_infos,
        "global_best_reward": float(global_best_reward),
        "global_best_info": global_best_info,
    }


def print_comparison_summary(
    args, run_dir, rl_results, bo_results, episode_plot_paths, summary_plot_paths
):
    rl_ep = rl_results["episode_best_rewards"]
    bo_ep = bo_results["episode_best_rewards"]

    rl_best = float(rl_results["global_best_reward"])
    bo_best = float(bo_results["global_best_reward"])

    if rl_best > bo_best:
        best_winner = f"RL ({args.model_type})"
    elif bo_best > rl_best:
        best_winner = "BayesOpt"
    else:
        best_winner = "Tie"

    if rl_ep.shape[0] != bo_ep.shape[0]:
        raise RuntimeError(
            "Episode count mismatch between RL and BayesOpt results. "
            f"RL={rl_ep.shape[0]}, BayesOpt={bo_ep.shape[0]}"
        )

    print("\n=== Comparison (Best Single-Step Reward Per Episode) ===")
    print(f"Episodes: {args.episodes} | Steps/episode: {args.max_actions}")
    print(f"Model checkpoint: {rl_results['model_path']}")

    print("\nEpisode-by-episode best-step reward comparison:")
    print("Episode | RL Reward | BayesOpt Reward | Delta (RL-BO) | Winner")

    rl_wins = 0
    bo_wins = 0
    ties = 0

    for ep_idx in range(rl_ep.shape[0]):
        rl_reward = float(rl_ep[ep_idx])
        bo_reward = float(bo_ep[ep_idx])
        delta = rl_reward - bo_reward

        if np.isclose(rl_reward, bo_reward):
            winner = "Tie"
            ties += 1
        elif rl_reward > bo_reward:
            winner = f"RL ({args.model_type})"
            rl_wins += 1
        else:
            winner = "BayesOpt"
            bo_wins += 1

        print(
            f"{ep_idx + 1:>7} | {rl_reward:>9.4f} | {bo_reward:>14.4f} | "
            f"{delta:>13.4f} | {winner}"
        )

    print("\nEpisode win count:")
    print(f"RL ({args.model_type}): {rl_wins}")
    print(f"BayesOpt: {bo_wins}")
    print(f"Ties: {ties}")
    print(f"Overall best RL reward: {rl_best:.4f}")
    print(f"Overall best BayesOpt reward: {bo_best:.4f}")
    print(f"Winner by overall best reward: {best_winner}")
    print(f"Saved per-episode waveform plots: {len(episode_plot_paths)}")
    if episode_plot_paths:
        print(
            f"Episode waveform plot directory: {os.path.dirname(episode_plot_paths[0])}"
        )
    print(f"Saved summary comparison plots: {len(summary_plot_paths)}")
    for p in summary_plot_paths:
        print(f"  {p}")
    print(f"Output directory: {run_dir}")


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        os.path.expanduser(args.output_dir),
        f"{timestamp}_{args.model_type}_{args.waveform_type}_{args.criterion_type}",
    )
    os.makedirs(run_dir, exist_ok=True)

    seeds = [args.seed + ep_idx for ep_idx in range(args.episodes)]

    print(
        "Running comparison with settings: "
        f"model_type={args.model_type}, waveform_type={args.waveform_type}, "
        f"criterion_type={args.criterion_type}, episodes={args.episodes}, "
        f"max_actions={args.max_actions}"
    )

    rl_results = evaluate_rl(args, seeds, run_dir)
    bo_results = evaluate_bayesopt(args, seeds, run_dir)
    summary_plot_paths = save_comparison_summary_plots(
        args, run_dir, rl_results, bo_results
    )
    episode_plot_paths = save_episode_waveform_comparison_plots(
        args, run_dir, rl_results, bo_results
    )

    print_comparison_summary(
        args,
        run_dir,
        rl_results,
        bo_results,
        episode_plot_paths,
        summary_plot_paths,
    )


if __name__ == "__main__":
    main()
