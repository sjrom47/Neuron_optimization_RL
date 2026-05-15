"""Stable-Baselines3 agent wrappers used by training scripts."""

from simulation.agents.ppo import PPOClass
from simulation.agents.recurrent_ppo import RecurrentPPOClass
from simulation.agents.sac import SACClass
from simulation.agents.td3 import TD3Class
from simulation.agents.tqc import TQCClass

AGENT_CLASSES = {
    "ppo": PPOClass,
    "recurrentppo": RecurrentPPOClass,
    "sac": SACClass,
    "td3": TD3Class,
    "tqc": TQCClass,
}


def get_agent_class(model_type: str):
    """Return the training wrapper class for a configured model type."""
    try:
        return AGENT_CLASSES[model_type.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_CLASSES))
        raise ValueError(f"Unsupported model type: {model_type}. Choose from: {supported}") from exc


__all__ = [
    "AGENT_CLASSES",
    "PPOClass",
    "RecurrentPPOClass",
    "SACClass",
    "TD3Class",
    "TQCClass",
    "get_agent_class",
]
