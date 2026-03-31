import random

from gymnasium import Env, spaces

from waveforms import FourierWaveform, Legendre3Waveform, SquareWaveform


class NEURONEnv(Env):
    def __init__(self, waveform_type, criterion, max_actions=10):
        super().__init__()
        self.waveform_type = waveform_type
        self.criterion = criterion
        self.waveform_type = waveform_type
        self.waveform = self.init_waveform()
        self.state = spaces.Dict(
            {
                "electrode_radius": spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=float
                ),  # TODO: change to actual bounds
                "angle_1": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
                "angle_2": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
                "neuron_type": spaces.Discrete(2),  # Assuming 2 neuron types
                # TODO: change actual bounds and types for these parameters
                "last_waveform_params": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.waveform.n_params,), dtype=float
                ),
                # TODO: change to actual bounds and types for these parameters
                "last_stimulation_params": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=float
                ),
            }
        )
        self.actions_taken = 0
        self.max_actions = max_actions

    def init_waveform(self, **kwargs):
        if self.waveform_type == "fourier":
            return FourierWaveform(**kwargs)
        elif self.waveform_type == "legendre3":
            return Legendre3Waveform(**kwargs)
        elif self.waveform_type == "square":
            return SquareWaveform(**kwargs)
        else:
            raise ValueError(f"Unsupported waveform type: {self.waveform_type}")

    def step(self, action):
        waveform = self.waveform.generate_waveform(
            **action, duration=1.0, sampling_rate=1000
        )
        response = self.simulate_neuron_response(waveform)
        reward = self.criterion(response)

        # TODO: update state based on response and/or action

        self.actions_taken += 1
        terminated = self.actions_taken >= self.max_actions
        truncated = False  # You can implement truncation logic if needed

        return self.state, reward, terminated, truncated, {}

    def simulate_neuron_response(self, waveform):
        # TODO: make a function that calls neuron behind the scenes
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # TODO: actual bounds do not reflect reality here
        self.state[0] = random.randrange(0.0, 1.0)  # Radius of electrode
        self.state[1] = random.randrange(0.0, 1.0)  # Angle 1
        self.state[2] = random.randrange(0.0, 1.0)  # Angle 2
        self.state[3] = random.choice(
            list(range(self.state["neuron_type"].n))
        )  # Neuron type

        # TODO: see neuron initialization (setting neuron parameters)
        default_waveform = self.waveform.default_stimulation()
        response = self.simulate_neuron_response(default_waveform)

        # TODO: transform response into something useful for the agent (e.g. firing rate, latency, etc.)
        # TODO: Add that info into the state representation

        # TODO: Add stimulation parameters into the state representation
        # TODO: see how to handle variable number of parameters for different waveforms (Fourier has more parameters than square for example)

        return self.state
