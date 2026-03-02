import numpy as np
import torch

# ==================== NumPy versions (for data generation with SciPy) ====================
def alpha_m_np(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m_np(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def dm_dt(m, t, V_func):
    """dm/dt = α(1-m) - βm"""
    V = V_func(t)
    return alpha_m_np(V) * (1 - m) - beta_m_np(V) * m

def voltage_protocol(t):
    if 5 < t < 25:
        return 0.0
    return -65.0

# ==================== PyTorch versions (for PINN training) ====================
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * torch.exp(-(V + 65.0) / 18.0)