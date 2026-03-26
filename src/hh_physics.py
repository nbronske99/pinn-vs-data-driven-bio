import numpy as np
import torch

# ==================== NumPy versions (for data generation with SciPy) ====================
def alpha_m_np(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m_np(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def dm_dt(m, t, V_func):
    """dm/dt = alpha(1-m) - beta*m"""
    V = V_func(t)
    return alpha_m_np(V) * (1.0 - m) - beta_m_np(V) * m

def voltage_protocol(t, V_rest=-65.0, V_step=0.0, t_start=5.0, t_end=25.0):
    """Step voltage protocol: rest -> depolarize -> rest."""
    if t_start < t < t_end:
        return V_step
    return V_rest

# ==================== PyTorch versions (for PINN training) ====================
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * torch.exp(-(V + 65.0) / 18.0)