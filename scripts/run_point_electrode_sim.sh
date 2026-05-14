#!/bin/bash
set -euo pipefail

uv run python -m simulation.experiments.point_electrode --num_electrode 1 --amplitude 0 --pulse_width 700 --period 1 --total_time 500
uv run python -m simulation.experiments.point_electrode --num_electrode 1 --amplitude 300 --pulse_width 1800 --period 300 --total_time 900
uv run python -m simulation.experiments.point_electrode --num_electrode 2 --amplitude 1000 --pulse_width 9000 --period 1000 --total_time 5000
uv run python -m simulation.experiments.point_electrode --num_electrode 4 --amplitude 6000 --pulse_width 30000 --period 6000 --total_time 17000
uv run python -m simulation.experiments.point_electrode --num_electrode 8 --amplitude 10000 --pulse_width 170000 --period 10000 --total_time 80000
uv run python -m simulation.experiments.point_electrode --num_electrode 16 --amplitude 100000 --pulse_width 500000 --period 100000 --total_time 300000
uv run python -m simulation.experiments.uniform_sim 0 300 0 300
uv run python -m simulation.experiments.plot_summary_point_electrode
