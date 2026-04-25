import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):
    """
    Maps state -> probability distribution over actions
    π(a|s)

    Parameters
    ----------
    state_dim : int
        Number of inputs/state variables.
        For CartPole this is 4.
    action_dim : int
        Number of possible actions.
        For CartPole this is 2: left or right.
    hidden_size : int
        Number of neurons in each hidden layer.
    """

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Input
        -----
        x : torch.Tensor
            State tensor.

        Output
        ------
        probs : torch.Tensor
            Probability distribution over actions.
        """
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return probs

# ============================================================
# VALUE NETWORK (CRITIC)
# ============================================================

class ValueNetwork(nn.Module):
    """
    Neural network for the critic V(s).

    Parameters
    ----------
    state_dim : int
        Number of inputs/state variables.
        For CartPole this is 4.
    hidden_size : int
        Number of neurons in each hidden layer.
    """

    def __init__(self, state_dim, hidden_size=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Input
        -----
        x : torch.Tensor
            State tensor.

        Output
        ------
        value : torch.Tensor
            Estimated state value V(s).
        """
        value = self.net(x)
        return value