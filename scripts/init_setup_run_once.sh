#!/bin/bash
set -euo pipefail

# Install Python dependencies with uv and compile NEURON mechanisms.
uv sync
uv run nrnivmodl src/simulation/neuron/assets/mechanisms
