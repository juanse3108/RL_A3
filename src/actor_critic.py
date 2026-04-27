import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from networks import PolicyNetwork, ValueNetwork
from utils import select_action, evaluate_policy


# ============================================================
# ACTOR-CRITIC TRAINING
# ============================================================

def train_actor_critic(
        n_steps=100000,
        eval_interval=2500,
        n_eval_episodes=5,
        actor_lr=1e-4,
        critic_lr=1e-4,
        gamma=0.99,
        hidden_size=64):
    """
    Train basic Actor-Critic on CartPole.

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

    state, _ = env.reset()
    step = 0

    while step < n_steps:
        if step % eval_interval == 0:
            mean_return = evaluate_policy(actor, eval_env, n_eval_episodes)
            eval_returns.append(mean_return)
            eval_steps.append(step)

        action, log_prob = select_action(actor, state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        value = critic(state_tensor)

        with torch.no_grad():
            next_value = critic(next_state_tensor)
            td_target = reward + gamma * next_value * (1 - int(done))

        td_error = td_target - value

        actor_loss = -log_prob * td_error.detach()
        critic_loss = mse_loss(value, td_target)

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if done:
            state, _ = env.reset()
        else:
            state = next_state

        step += 1

    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_steps)


if __name__ == "__main__":
    returns, steps = train_actor_critic(n_steps=10000)
    print("Actor-Critic test finished.")
    print("Evaluations:", len(returns))
