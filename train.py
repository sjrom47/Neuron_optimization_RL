import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv

from environment import NEURONEnv
from models import RecurrentPPOClass, SACClass, TD3Class, TQCClass


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
        "--timesteps", type=int, default=1000, help="Number of training timesteps"
    )
    return parser.parse_args()


def make_env(waveform_type, criterion_type):
    def _init():
        return NEURONEnv(
            waveform_type=waveform_type, criterion_type=criterion_type, max_actions=10
        )

    return _init


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    env = SubprocVecEnv(
        [make_env(args.waveform_type, args.criterion_type) for _ in range(4)]
    )
    # TODO: maybe refactor into a factory at some point
    if args.model_type == "recurrentppo":
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
    )
    model.train()
    model.eval()
