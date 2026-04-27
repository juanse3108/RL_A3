# Reinforcement Learning Assignment 3

This repository contains implementations of three policy-gradient methods applied to the CartPole-v1 environment:

- REINFORCE  
- Actor-Critic (AC)  
- Advantage Actor-Critic (A2C)  

All methods use the same policy network architecture to ensure a fair comparison.

---

## Installation

Create and activate a virtual environment:

python -m venv rl_env
source rl_env/bin/activate

Install the required dependencies:

pip install -r requirements.txt

---

## Running the Experiments

To run all algorithms and generate the comparison plot:

python src/train.py

This script will:

- Train REINFORCE, Actor-Critic, and A2C
- Evaluate performance periodically
- Average results over multiple repetitions
- Save the final comparison plot in:

results/comparison.png

---

## Running on ALICE (HPC)

To run the experiments on the ALICE cluster:

sbatch run_experiment.sh

Check the job status:

squeue -u <your-username>

View logs:

tail -f logs/<job_id>.out

---

## Repository Structure

```text
RL_A3/
├── src/
│   ├── networks.py
│   ├── utils.py
│   ├── reinforce.py
│   ├── actor_critic.py
│   ├── a2c.py
│   └── train.py
├── results/
│   └── comparison.png
├── BaselineDataCartPole.csv
├── run_experiment.sh
├── requirements.txt
└── README.md
```
---

## Notes

- Each method is trained for 1,000,000 environment steps
- Results are averaged over multiple runs for stability
- A baseline from Assignment 2 is included for comparison