import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

from networks import PolicyNetwork
from utils import select_action, evaluate_policy


# ============================================================
# RETURN COMPUTATION
# ============================================================

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns G_t for one episode.

    Parameters
    ----------
    rewards : list[float]
        Rewards collected during one episode.
    gamma : float
        Discount factor for future rewards.

    Returns
    -------
    torch.Tensor
        Discounted return for each timestep.
    """

    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)

    # Normalize for more stable learning
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


# ============================================================
# REINFORCE TRAINING
# ============================================================

def train_reinforce(
        n_steps=100000,
        eval_interval=2500,
        n_eval_episodes=5,
        learning_rate=1e-3,
        gamma=0.99,
        hidden_size=64):
    """
    Train REINFORCE on CartPole.

    Parameters
    ----------
    n_steps : int
        Total number of environment steps used for training.
    eval_interval : int
        Number of environment steps between evaluations.
    n_eval_episodes : int
        Number of episodes used during evaluation.
    learning_rate : float
        Learning rate for the policy optimizer.
    gamma : float
        Discount factor for future rewards.
    hidden_size : int
        Number of neurons in each hidden layer.

    Returns
    -------
    eval_returns : np.ndarray
        Mean evaluation returns collected during training.
    eval_steps : np.ndarray
        Environment steps where evaluation was performed.
    """

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size
    )

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    eval_returns = []
    eval_steps = []

    step = 0

    while step < n_steps:
        state, _ = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done and step < n_steps: #episode loop
            if step % eval_interval == 0:
                mean_return = evaluate_policy(
                    policy,
                    eval_env,
                    n_episodes=n_eval_episodes
                )
                eval_returns.append(mean_return)
                eval_steps.append(step)

            action, log_prob = select_action(policy, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            step += 1

        returns = compute_returns(rewards, gamma)

        policy_loss = []

        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_steps)


if __name__ == "__main__":
    returns, steps = train_reinforce(n_steps=10000)
    print("REINFORCE test finished.")
    print("Evaluations:", len(returns))
