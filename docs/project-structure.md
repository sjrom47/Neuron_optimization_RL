# Project Structure

The code is organized as a Python package under `src/simulation`. New reusable code should live in this package instead of in the repository root.

## Core Modules

- `simulation.environment` defines the Gymnasium environment used by RL agents.
- `simulation.config` contains default NEURON and stimulation parameters.
- `simulation.callbacks` contains Stable-Baselines3 callbacks for plots, diagnostics, and waveform snapshots.
- `simulation.paths` centralizes filesystem paths for outputs and NEURON assets.

## Subpackages

- `simulation.agents` contains wrappers for PPO, RecurrentPPO, SAC, TD3, and TQC. Add new training wrappers here and register them in `simulation.agents.AGENT_CLASSES`.
- `simulation.criteria` contains reward functions. Each criterion should inherit from `Criterion`.
- `simulation.waveforms` contains waveform parameterizations. Each waveform should inherit from `Waveform` and expose `n_params` and `param_bounds`.
- `simulation.fields` contains electric-field implementations and sparse-placement utilities.
- `simulation.neuron` contains the NEURON integration layer. Runtime HOC, cell, and MOD assets live under `simulation.neuron.assets`.
- `simulation.training` contains command-line training and evaluation entry points.
- `simulation.experiments` contains exploratory or paper-reproduction scripts that are useful but not part of the stable core API.

## Adding New Components

To add a waveform:

1. Create a module in `src/simulation/waveforms/`.
2. Implement the `Waveform` interface.
3. Export it from `src/simulation/waveforms/__init__.py`.
4. Register it in `NEURONEnv.init_waveform`.

To add a reward criterion:

1. Create a module in `src/simulation/criteria/`.
2. Implement the `Criterion` interface.
3. Export it from `src/simulation/criteria/__init__.py`.
4. Register it in `NEURONEnv.init_criterion`.

To add an agent:

1. Create a wrapper in `src/simulation/agents/`.
2. Match the existing wrapper interface: constructor, `train()`, and `eval()`.
3. Register it in `AGENT_CLASSES`.

## Outputs

Training and evaluation write generated data to root-level output directories such as `plots/`, `weights/`, and `TISimResults/`. These are intentionally kept outside `src/` and ignored by git.
