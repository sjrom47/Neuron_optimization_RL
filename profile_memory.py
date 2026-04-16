import os
import psutil

os.environ["NEURON_MODULE_OPTIONS"] = "-nogui -NSTACK 100000 -NFRAME 20000"
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

proc = psutil.Process()

def rss_mb():
    return proc.memory_info().rss / 1024 ** 2

def checkpoint(label, prev):
    current = rss_mb()
    print(f"[{current:8.1f} MB] (+{current - prev:7.1f} MB)  {label}")
    return current

mem = rss_mb()
print(f"[{mem:8.1f} MB] baseline (python started)")

# ── numpy / scipy ──────────────────────────────────────────────────────────────
import numpy as np
from scipy.signal import find_peaks
mem = checkpoint("numpy + scipy imported", mem)

# ── torch / stable-baselines3 ─────────────────────────────────────────────────
import torch
mem = checkpoint("torch imported", mem)

from stable_baselines3 import TD3
mem = checkpoint("stable_baselines3 imported", mem)

# ── NEURON ────────────────────────────────────────────────────────────────────
from neuron import h, gui, rxd
from neuron.units import ms, mV
mem = checkpoint("neuron imported", mem)

# ── .hoc files ────────────────────────────────────────────────────────────────
h.load_file("stdrun.hoc")
h.load_file("nrngui.hoc")
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")
h.load_file("interpCoordinates.hoc")
h.load_file("setPointers.hoc")
h.load_file("cellChooser.hoc")
h.load_file("setParams.hoc")
h.load_file("editMorphology.hoc")
mem = checkpoint(".hoc files loaded", mem)

# ── cell model ────────────────────────────────────────────────────────────────
h('setParamsAdultHuman()')
h('cell_chooser(36)')
h.celsius = 37.0
h.dt = 0.1
mem = checkpoint("cell_chooser(36) called", mem)

# ── extracellular / xtra mechanisms ──────────────────────────────────────────
import sys
sys.path.insert(0, os.getcwd())
from neuron_model_serial import NeuronSim
from elec_field import ICMS
mem = checkpoint("NeuronSim / ICMS imported", mem)

elec_field = ICMS(x=0, y=1, z=0, conductivity=0.33)
neuron = NeuronSim(human_or_mice=0, cell_id=36, temp=37.0, dt=0.1, elec_field=elec_field)
mem = checkpoint("NeuronSim instantiated (cell loaded)", mem)

neuron._set_xtra_param(angle=np.array([0, 0]), pos_neuron=np.array([0, 0, 0]))
mem = checkpoint("_set_xtra_param called", mem)

# ── save-state / first stimulate ──────────────────────────────────────────────
sampling_rate = 1e5
delay_init = 2000
delay_final = 5
stimulation_duration = 30  # ms
n_samples = int(stimulation_duration * sampling_rate * 1e-3)
time_array = np.arange(n_samples) / sampling_rate * 1000
amp_array = np.zeros(n_samples)

neuron.stimulate(
    time_array=time_array,
    amp_array=amp_array,
    amp_array2=amp_array.copy(),
    sampling_rate=sampling_rate,
    delay_init=delay_init,
    delay_final=delay_final,
)
mem = checkpoint("stimulate() call #1", mem)

# ── repeated stimulate calls ──────────────────────────────────────────────────
for i in range(2, 6):
    neuron.stimulate(
        time_array=time_array,
        amp_array=amp_array,
        amp_array2=amp_array.copy(),
        sampling_rate=sampling_rate,
        delay_init=delay_init,
        delay_final=delay_final,
    )
    mem = checkpoint(f"stimulate() call #{i}", mem)

# ── sb3_contrib / RecurrentPPO ────────────────────────────────────────────────
from sb3_contrib import RecurrentPPO
mem = checkpoint("sb3_contrib RecurrentPPO imported", mem)

from environment import NEURONEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return NEURONEnv(waveform_type="fourier", criterion_type="min_energy", max_actions=10)

env = DummyVecEnv([make_env])
mem = checkpoint("DummyVecEnv created", mem)

env.reset()
mem = checkpoint("env.reset() called", mem)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=1e-4,
    verbose=0,
    n_steps=200,
    batch_size=50,
)
mem = checkpoint("RecurrentPPO model created", mem)

# Run a handful of training steps
model.learn(total_timesteps=50)
mem = checkpoint("model.learn(50 steps)", mem)

model.learn(total_timesteps=200)
mem = checkpoint("model.learn(200 more steps)", mem)

model.learn(total_timesteps=500)
mem = checkpoint("model.learn(500 more steps)", mem)

print("\nDone.")
