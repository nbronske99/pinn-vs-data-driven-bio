import torch
import torch.nn as nn
from .hh_physics import alpha_m, beta_m

class DataDrivenNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_residual(self, x):
        """Compute ODE residual: dm/dt - [α(1-m) - βm]"""
        x = x.clone().requires_grad_(True)
        m_pred = self(x)
        dm_dt = torch.autograd.grad(
            m_pred, x, grad_outputs=torch.ones_like(m_pred),
            create_graph=True
        )[0][:, 0:1]
        V = x[:, 1:2]
        return dm_dt - (alpha_m(V) * (1 - m_pred) - beta_m(V) * m_pred)