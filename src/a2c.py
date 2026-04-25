import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from networks import PolicyNetwork, ValueNetwork
from utils import select_action, evaluate_policy
from reinforce import compute_returns


# ============================================================
# ADVANTAGE ACTOR-CRITIC TRAINING
# ============================================================

def train_a2c(
        n_steps=100000,
        eval_interval=2500,
        n_eval_episodes=5,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        hidden_size=64):
    """
    Train Advantage Actor-Critic (A2C) on CartPole.

    Parameters
    ----------
    n_steps : int
        Total number of environment steps used for training.
    eval_interval : int
        Number of environment steps between evaluations.
    n_eval_episodes : int
        Number of episodes used during evaluation.
    actor_lr : float
        Learning rate for the policy network.
    critic_lr : float
        Learning rate for the value network.
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

    actor = PolicyNetwork(state_dim, action_dim, hidden_size)
    critic = ValueNetwork(state_dim, hidden_size)

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    mse_loss = nn.MSELoss()

    eval_returns = []
    eval_steps = []

    step = 0

    while step < n_steps:
        state, _ = env.reset()
        done = False

        states = []
        log_probs = []
        rewards = []

        while not done and step < n_steps:
            if step % eval_interval == 0:
                mean_return = evaluate_policy(
                    actor,
                    eval_env,
                    n_episodes=n_eval_episodes
                )
                eval_returns.append(mean_return)
                eval_steps.append(step)

            action, log_prob = select_action(actor, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            step += 1

        returns = compute_returns(rewards, gamma)

        states_tensor = torch.FloatTensor(np.array(states))
        values = critic(states_tensor).squeeze(1)

        advantages = returns - values.detach()

        actor_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            actor_loss.append(-log_prob * advantage)

        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = mse_loss(values, returns)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_steps)


if __name__ == "__main__":
    returns, steps = train_a2c(n_steps=10000)
    print("A2C test finished.")
    print("Evaluations:", len(returns))
