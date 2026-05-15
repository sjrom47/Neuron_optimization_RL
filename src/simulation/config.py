"""Default simulation and stimulation settings."""

# NEURON simulation
NEURON_HUMAN_OR_MICE = 0
NEURON_TEMP = 37.0  # °C
NEURON_DT = 0.1  # ms
NEURON_TYPES = [36, 6]

# Stimulation timing
STIMULATION_DURATION = 30  # ms
DELAY_INIT = 2000  # pre-stimulus settling time
DELAY_FINAL = 5  # post-stimulus recording time
SAMPLING_RATE = 1e5  # Hz

# Electrode position perturbation
PERTURBATE_ELECTRODE_POSITION = False
ELECTRODE_POSITION_PERTURBATION_SIGMA = 0.01  # mm

# Multiple neuron types
MULTPLE_NEURON_TYPES = True
