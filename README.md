# Policy gradient RL for waveform optimization using the NEURON simulator

This repository contains code for a project on policy gradient reinforcement learning for optimizing stimulation waveforms using the NEURON simulator. The code is structured as follows:
- `config.py`: Contains configuration parameters for the environment and training.
- `environment.py`: Contains the implementation of the environment, including the state representation, action space, and reward function.
- `train.py`: Contains the main training loop for the agent.
- `requirements.txt`: Contains the required Python packages for running the code.
Then there are several components with different versions that each have their own directory:
- `criterions/`: includes a base abstract class for the reward criterion and two implementations: `MinEnergy` and `SelectivityCriterion`.
- `models/`: includes the classes for the four different RL agents used and the necessary callbacks to create logs and plots during training.
- `waveforms/`:  contains a base abstract class for the waveform and many concrete implementations of different waveforms

### Running the code: 
1. Install uv on your system (linux or WSL), run uv sync in the repo directory and then run the second command in the setup sh. As an alternative, you can also install the required packages using pip install -r requirements.txt. Additional installation steps may be necessary to install neuron in windows machines, so refer to the official neuron installation guide.
2. Run the following command:
```bash
nrnivmodl mechanisms
```
Alternatively if you are using pip to install the dependencies, you can run the following command to compile the neuron mechanisms:
```bash
bash init_setup_run_once.sh
```
Which will perform steps 1 and 2 in one go.


### Training the agent:
```bash
python train.py
```

This will start the training process and save the trained model. It will use TD3 as the default agent. To change the agent and modify other training parameters such as the reward criterion, the number of episodes, the learning rate, etc. consult both the `config.py` file and the `train.py` arguments. 

### Comparing against bayesian optimization:

To compare the performance of the RL agent against a bayesian optimization approach, run the following command:
```bash
python compare_vs_bayesopt_baseline.py
```
This will compare the model of the RL agent against a bayesian optimization baseline and show the comparative results





