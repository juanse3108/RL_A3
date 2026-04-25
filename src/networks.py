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
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return probs

# ============================================================
# VALUE NETWORK (CRITIC)
# ============================================================

class ValueNetwork(nn.Module):
    """
    Maps state -> scalar value V(s)
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
        value = self.net(x)
        return value