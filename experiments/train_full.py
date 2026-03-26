"""
Step 3: Train both models on the full training set (800 points).

Produces detailed training curves and full-domain predictions
for both models. Used by the analysis script for comparison plots.

Usage: python experiments/train_full.py
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


def main():
    with open("configs/experiment.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data = np.load(cfg["data"]["save_path"])
    t = data["t"]
    V = data["V"]
    m_true = data["m_true"]

    X_all = np.stack([t, V], axis=1)

    np.random.seed(42)
    n = len(t)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
    y_train = torch.tensor(m_true[train_idx], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
    y_test = torch.tensor(m_true[test_idx], dtype=torch.float32).unsqueeze(1)
    X_full = torch.tensor(X_all, dtype=torch.float32)

    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples:     {len(test_idx)}")

    results_dir = cfg["paths"]["data_results"]
    os.makedirs(results_dir, exist_ok=True)

    criterion = nn.MSELoss()

    # ==================== DATA-DRIVEN MODEL ====================
    print("\nTraining data-driven baseline...")
    torch.manual_seed(0)
    dd_model = DataDrivenNN(hidden_dim=cfg["model"]["hidden_dim"])
    dd_optimizer = optim.Adam(dd_model.parameters(), lr=cfg["data_driven"]["learning_rate"])

    dd_train_losses = []
    dd_test_losses = []

    for epoch in range(cfg["data_driven"]["epochs"]):
        dd_optimizer.zero_grad()
        pred = dd_model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        dd_optimizer.step()

        dd_train_losses.append(loss.item())
        with torch.no_grad():
            dd_test_losses.append(criterion(dd_model(X_test), y_test).item())

        if epoch % 400 == 0:
            print(f"  Epoch {epoch:4d} | Train: {dd_train_losses[-1]:.6f} | Test: {dd_test_losses[-1]:.6f}")

    dd_pred = dd_model(X_full).detach().numpy().flatten()

    # ==================== PINN ====================
    print("\nTraining PINN...")
    torch.manual_seed(0)
    pinn_model = PINN(hidden_dim=cfg["model"]["hidden_dim"])
    pinn_optimizer = optim.Adam(pinn_model.parameters(), lr=cfg["pinn"]["learning_rate"])
    lambda_phys = cfg["pinn"]["lambda_phys"]

    pinn_train_losses = []
    pinn_test_losses = []
    pinn_data_losses = []
    pinn_phys_losses = []

    X_colloc = X_full.clone()

    for epoch in range(cfg["pinn"]["epochs"]):
        pinn_optimizer.zero_grad()
        m_pred_train = pinn_model(X_train)
        data_loss = criterion(m_pred_train, y_train)
        phys_res = pinn_model.physics_residual(X_colloc)
        phys_loss = torch.mean(phys_res ** 2)
        loss = data_loss + lambda_phys * phys_loss
        loss.backward()
        pinn_optimizer.step()

        pinn_train_losses.append(loss.item())
        pinn_data_losses.append(data_loss.item())
        pinn_phys_losses.append(phys_loss.item())
        with torch.no_grad():
            pinn_test_losses.append(criterion(pinn_model(X_test), y_test).item())

        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | Total: {loss.item():.6f} | Data: {data_loss.item():.6f} | Phys: {phys_loss.item():.6f} | Test: {pinn_test_losses[-1]:.6f}")

    pinn_pred = pinn_model(X_full).detach().numpy().flatten()

    # ==================== SAVE ====================
    full_results = {
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    with open(os.path.join(results_dir, "full_train_meta.json"), "w") as f:
        json.dump(full_results, f)

    np.savez(
        os.path.join(results_dir, "full_train_results.npz"),
        dd_train_losses=np.array(dd_train_losses),
        dd_test_losses=np.array(dd_test_losses),
        dd_pred=dd_pred,
        pinn_train_losses=np.array(pinn_train_losses),
        pinn_test_losses=np.array(pinn_test_losses),
        pinn_data_losses=np.array(pinn_data_losses),
        pinn_phys_losses=np.array(pinn_phys_losses),
        pinn_pred=pinn_pred,
    )

    torch.save(dd_model.state_dict(), os.path.join(results_dir, "dd_weights.pt"))
    torch.save(pinn_model.state_dict(), os.path.join(results_dir, "pinn_weights.pt"))

    print(f"\nResults saved to {results_dir}/")
    print(f"  DD  final test MSE: {dd_test_losses[-1]:.6f}")
    print(f"  PINN final test MSE: {pinn_test_losses[-1]:.6f}")


if __name__ == "__main__":
    main()