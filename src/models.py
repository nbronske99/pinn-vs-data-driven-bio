import torch
import torch.nn as nn
from .hh_physics import alpha_m, beta_m


class DataDrivenNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

    def physics_residual(self, x):
        """Compute ODE residual: dm/dt - [alpha(V)(1-m) - beta(V)m]"""
        x = x.clone().requires_grad_(True)
        m_pred = self(x)
        dm_dt = torch.autograd.grad(
            m_pred, x,
            grad_outputs=torch.ones_like(m_pred),
            create_graph=True,
        )[0][:, 0:1]
        V = x[:, 1:2]
        return dm_dt - (alpha_m(V) * (1.0 - m_pred) - beta_m(V) * m_pred)