# Reinforcement Learning Assignment 3

This repository contains implementations of three policy-gradient methods on CartPole-v1:

- REINFORCE
- Basic Actor-Critic
- Advantage Actor-Critic (A2C)

All three methods use the same policy network architecture.

## Project Structure

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
