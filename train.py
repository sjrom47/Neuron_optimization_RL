from environment import NEURONEnv
from models import RecurrentPPOClass, SACClass, TD3Class, TQCClass


def parse_args():
    import argparse

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
        "--timesteps", type=int, default=10000, help="Number of training timesteps"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    env = NEURONEnv(
        waveform_type=args.waveform_type, criterion_type=args.criterion_type
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
        env, env.waveform, env.criterion, lr=args.lr, timesteps=args.timesteps
    )
    model.train()
    model.eval()
