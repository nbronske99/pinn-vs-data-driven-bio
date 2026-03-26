"""
Step 2: Run the core ablation experiment.

Trains both data-driven and PINN models across multiple training set sizes,
with multiple random seeds per condition. Records test MSE for each.

Usage: python experiments/run_ablation.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import DataDrivenNN, PINN


def train_data_driven(X_train, y_train, X_test, y_test, cfg, seed=0):
    torch.manual_seed(seed)
    model = DataDrivenNN(hidden_dim=cfg["model"]["hidden_dim"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["data_driven"]["learning_rate"])
    criterion = nn.MSELoss()

    for epoch in range(cfg["data_driven"]["epochs"]):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_mse = criterion(model(X_test), y_test).item()
    return test_mse, model


def train_pinn(X_train, y_train, X_test, y_test, X_colloc, cfg, seed=0):
    torch.manual_seed(seed)
    model = PINN(hidden_dim=cfg["model"]["hidden_dim"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["pinn"]["learning_rate"])
    criterion = nn.MSELoss()
    lambda_phys = cfg["pinn"]["lambda_phys"]

    for epoch in range(cfg["pinn"]["epochs"]):
        optimizer.zero_grad()
        data_loss = criterion(model(X_train), y_train)
        phys_res = model.physics_residual(X_colloc)
        phys_loss = torch.mean(phys_res ** 2)
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_mse = criterion(model(X_test), y_test).item()
    return test_mse, model


def main():
    with open("configs/experiment.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data = np.load(cfg["data"]["save_path"])
    t = data["t"]
    V = data["V"]
    m_true = data["m_true"]

    X_all = np.stack([t, V], axis=1)

    ab = cfg["ablation"]
    np.random.seed(42)
    all_idx = np.random.permutation(len(t))
    test_idx = all_idx[: ab["n_test"]]
    pool_idx = all_idx[ab["n_test"] :]

    X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
    y_test = torch.tensor(m_true[test_idx], dtype=torch.float32).unsqueeze(1)
    X_full = torch.tensor(X_all, dtype=torch.float32)

    print(f"Fixed test set: {len(test_idx)} points")
    print(f"Training pool:  {len(pool_idx)} points")

    results = {
        "train_sizes": ab["train_sizes"],
        "n_trials": ab["n_trials"],
        "dd": {},
        "pinn": {},
    }

    for n_train in ab["train_sizes"]:
        print(f"\n===== Training size: {n_train} =====")
        dd_mses = []
        pinn_mses = []

        for trial, seed in enumerate(ab["trial_seeds"]):
            rng = np.random.RandomState(seed * 100 + n_train)
            sub_idx = rng.choice(pool_idx, size=n_train, replace=False)

            X_train = torch.tensor(X_all[sub_idx], dtype=torch.float32)
            y_train = torch.tensor(m_true[sub_idx], dtype=torch.float32).unsqueeze(1)

            dd_mse, _ = train_data_driven(X_train, y_train, X_test, y_test, cfg, seed=seed)
            dd_mses.append(dd_mse)

            pinn_mse, _ = train_pinn(X_train, y_train, X_test, y_test, X_full, cfg, seed=seed)
            pinn_mses.append(pinn_mse)

            print(f"  Trial {trial+1}: DD={dd_mse:.6f}  PINN={pinn_mse:.6f}")

        results["dd"][str(n_train)] = {
            "mses": dd_mses,
            "mean": float(np.mean(dd_mses)),
            "std": float(np.std(dd_mses)),
        }
        results["pinn"][str(n_train)] = {
            "mses": pinn_mses,
            "mean": float(np.mean(pinn_mses)),
            "std": float(np.std(pinn_mses)),
        }

    results_path = os.path.join(cfg["paths"]["data_results"], "ablation_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    print(f"\n{'N_train':>8}  {'DD MSE':>14}  {'PINN MSE':>14}  {'Improvement':>12}")
    print("-" * 54)
    for n in ab["train_sizes"]:
        dd_mean = results["dd"][str(n)]["mean"]
        dd_std = results["dd"][str(n)]["std"]
        pinn_mean = results["pinn"][str(n)]["mean"]
        pinn_std = results["pinn"][str(n)]["std"]
        improv = (dd_mean - pinn_mean) / dd_mean * 100 if dd_mean > 0 else 0
        print(f"{n:>8d}  {dd_mean:.6f}+/-{dd_std:.4f}  {pinn_mean:.6f}+/-{pinn_std:.4f}  {improv:>+10.1f}%")


if __name__ == "__main__":
    main()