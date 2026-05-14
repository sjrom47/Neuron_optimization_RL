# NEURON Assets

NEURON-related runtime assets are stored under `src/simulation/neuron/assets/`:

```text
assets/
  hoc/         Shared HOC setup files
  cells/       Cell templates, morphology, biophysics, and save states
  mechanisms/  MOD mechanism files compiled by nrnivmodl
```

## Mechanism Compilation

Compile mechanisms from the repository root:

```bash
uv run nrnivmodl src/simulation/neuron/assets/mechanisms
```

This creates platform-specific generated directories such as `x86_64/` or `arm64/`. They are build artifacts and should not be committed.

Run `uv sync` before compiling so the NEURON package and the pinned `setuptools<81` build tooling are available in the project environment.

## HOC Loading

`simulation.neuron.model.NeuronSim` loads shared HOC files by absolute path through `simulation.paths.HOC_DIR`. It also changes NEURON's working directory to `simulation.paths.NEURON_ASSETS_DIR` before loading `cellChooser.hoc`, because that HOC file loads cell-specific files through paths relative to the asset root.

## Save States

NEURON burn-in save states are written to:

```text
src/simulation/neuron/assets/cells/SaveState/
```

These files are generated at runtime and can be recreated. They should be treated as cache files unless you intentionally need to preserve a specific simulation state.

## Troubleshooting

If NEURON cannot find a mechanism, re-run `uv run nrnivmodl src/simulation/neuron/assets/mechanisms` from the repository root.

If HOC loading fails after moving files, check that `simulation.paths.NEURON_ASSETS_DIR` points to `src/simulation/neuron/assets` and that the expected cell folder exists under `assets/cells/`.

If training starts many worker processes, remember that each Ray actor owns a NEURON instance. Memory use can grow quickly with the number of vectorized environments and neuron types.
