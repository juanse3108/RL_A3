import csv
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# ACTION SAMPLING
# ============================================================

def select_action(policy_network, state):
    """
    Sample an action from the policy network.

    Parameters
    ----------
    policy_network : PolicyNetwork
        Network that maps state -> action probabilities.
    state : np.ndarray
        Current environment state.

    Returns
    -------
    action : int
        Sampled action index.
    log_prob : torch.Tensor
        Log probability of the sampled action.
    """

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_network(state_tensor)
    dist = torch.distributions.Categorical(probs)

    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(policy_network, eval_env, n_episodes=5):
    """
    Evaluate policy without exploration noise beyond policy sampling.

    Parameters
    ----------
    policy_network : PolicyNetwork
        Trained policy network.
    eval_env : gymnasium.Env
        Environment used only for evaluation.
    n_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    float
        Mean return over evaluation episodes.
    """

    returns = []

    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            #action, _ = select_action(policy_network, state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = policy_network(state_tensor)
            action = torch.argmax(probs, dim=-1).item()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)

    return np.mean(returns)


# ============================================================
# SMOOTHING
# ============================================================

def smooth(values, window=9):
    """
    Smooth a learning curve with moving average.

    Parameters
    ----------
    values : np.ndarray
        Raw return values.
    window : int
        Moving average window size.

    Returns
    -------
    np.ndarray
        Smoothed values.
    """

    return np.convolve(values, np.ones(window) / window, mode="valid")


# ============================================================
# BASELINE LOADING
# ============================================================

def load_baseline(csv_path="BaselineDataCartPole.csv"):
    """
    Load Assignment 2 baseline curve.

    Parameters
    ----------
    csv_path : str
        Path to baseline CSV file.

    Returns
    -------
    steps : np.ndarray
        Environment steps.
    returns : np.ndarray
        Smoothed episode returns.
    """

    steps, returns = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.append(float(row["env_step"]))
            returns.append(float(row["Episode_Return_smooth"]))

    return np.array(steps), np.array(returns)


# ============================================================
# PLOTTING
# ============================================================

class LearningCurvePlot:
    """
    Helper class for plotting learning curves.
    """

    def __init__(self, title="Learning Curve"):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Environment Steps")
        self.ax.set_ylabel("Episode Return")
        self.ax.set_title(title)

    def add_curve(self, steps, returns, label):
        self.ax.plot(steps, returns, label=label)

    def set_ylim(self, low, high):
        self.ax.set_ylim(low, high)

    def save(self, filename):
        self.ax.legend()
        self.fig.savefig(filename)
        plt.close(self.fig)
        print(f"Saved: {filename}")
