import argparse

import ray
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from environment import NEURONEnv
from models import PPOClass, RecurrentPPOClass, SACClass, TD3Class, TQCClass


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train RL model for neuron stimulation"
    )
    parser.add_argument(
        "--waveform_type", type=str, default="fourier", help="Type of waveform to use"
    )
    parser.add_argument(
        "--criterion_type",
        type=str,
        default="min_energy",
        help="Type of criterion to optimize",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="recurrentppo",
        help="Type of RL model to train",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--normalize_obs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable observation normalization with VecNormalize (disabled by default)",
    )
    parser.add_argument(
        "--max_amplitude",
        type=float,
        default=1000.0,
        help="Max waveform amplitude (mA)",
    )
    return parser.parse_args()


def make_env(waveform_type, criterion_type, max_amplitude=500.0):
    def _init():
        return NEURONEnv(
            waveform_type=waveform_type,
            criterion_type=criterion_type,
            max_actions=10,
            max_amplitude=max_amplitude,
        )

    return _init


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    # Start a single local Ray head here so every SubprocVecEnv worker can
    # attach to it via address="auto" instead of each spinning up its own.
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    # env = DummyVecEnv([make_env(args.waveform_type, args.criterion_type)])
    # print('env created')
    # Attempt #1 to increase speed: Parallelization of environments
    num_envs = 8
    envs = []
    for i in range(num_envs):
        envs.append(
            make_env(args.waveform_type, args.criterion_type, args.max_amplitude)
        )
    env = SubprocVecEnv(envs, start_method="spawn")
    cell_id = env.get_attr("neuron_types")[0][0]
    if args.normalize_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # TODO: maybe refactor into a factory at some point
    if args.model_type == "ppo":
        model_class = PPOClass
    elif args.model_type == "recurrentppo":
        model_class = RecurrentPPOClass
    elif args.model_type == "sac":
        model_class = SACClass
    elif args.model_type == "td3":
        model_class = TD3Class
    elif args.model_type == "tqc":
        model_class = TQCClass
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = model_class(
        env,
        args.waveform_type,
        args.criterion_type,
        lr=args.lr,
        timesteps=args.timesteps,
        cell_id=cell_id,
    )
    model.train()
    if args.normalize_obs and hasattr(model.env, "training"):
        model.env.training = False
        model.env.norm_reward = False
    model.eval()
