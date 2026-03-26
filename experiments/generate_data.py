"""
Step 1: Generate synthetic data from the Hodgkin-Huxley gating ODE.

Solves dm/dt = alpha(V)(1-m) - beta(V)m using SciPy's odeint
under a step voltage protocol. Saves the ground truth to data/raw/.

Usage: python experiments/generate_data.py
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hh_physics import dm_dt, voltage_protocol


def main():
    with open("configs/experiment.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]

    t = np.linspace(dc["t_start"], dc["t_end"], dc["n_points"])

    def V_func(ti):
        return voltage_protocol(
            ti,
            V_rest=dc["voltage_rest"],
            V_step=dc["voltage_step"],
            t_start=dc["step_start"],
            t_end=dc["step_end"],
        )

    V_array = np.array([V_func(ti) for ti in t])

    m0 = [dc["m_initial"]]
    m_true = odeint(dm_dt, m0, t, args=(V_func,))[:, 0]

    os.makedirs(os.path.dirname(dc["save_path"]), exist_ok=True)
    np.savez(dc["save_path"], t=t, V=V_array, m_true=m_true)

    print(f"Synthetic data saved to {dc['save_path']}")
    print(f"  Time points: {len(t)}")
    print(f"  Voltage range: {V_array.min():.1f} to {V_array.max():.1f} mV")
    print(f"  m_true range:  {m_true.min():.3f} to {m_true.max():.3f}")


if __name__ == "__main__":
    main()