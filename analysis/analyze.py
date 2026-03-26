"""
Step 4: Analyze results and generate figures.

Reads raw experiment outputs from data/results/ and produces:
  1. Ablation plot (MSE vs. training size, log-log)
  2. Full-data prediction overlay
  3. Pointwise error comparison
  4. Training dynamics comparison
  5. Metrics summary (printed to stdout)

Usage: python analysis/analyze.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_metrics(y_true, y_pred, idx):
    err = y_true[idx] - y_pred[idx]
    return {
        "MSE": float(np.mean(err ** 2)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Max Error": float(np.max(np.abs(err))),
    }


def main():
    with open("configs/experiment.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    fig_dir = cfg["paths"]["figures"]
    os.makedirs(fig_dir, exist_ok=True)

    data = np.load(cfg["data"]["save_path"])
    t = data["t"]
    V = data["V"]
    m_true = data["m_true"]

    # ==================== ABLATION PLOT ====================
    ablation_path = os.path.join(cfg["paths"]["data_results"], "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            ab = json.load(f)

        sizes = ab["train_sizes"]
        dd_means = [ab["dd"][str(n)]["mean"] for n in sizes]
        dd_stds = [ab["dd"][str(n)]["std"] for n in sizes]
        pinn_means = [ab["pinn"][str(n)]["mean"] for n in sizes]
        pinn_stds = [ab["pinn"][str(n)]["std"] for n in sizes]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(sizes, dd_means, yerr=dd_stds,
                    marker="o", capsize=5, lw=2, ms=8,
                    label="Data-Driven NN (ReLU)", color="#e74c3c")
        ax.errorbar(sizes, pinn_means, yerr=pinn_stds,
                    marker="s", capsize=5, lw=2, ms=8,
                    label="PINN (Tanh + ODE constraint)", color="#2ecc71")
        ax.set_xlabel("Number of Training Points", fontsize=13)
        ax.set_ylabel("Test MSE", fontsize=13)
        ax.set_title("Generalization Error vs. Training Set Size\n(Hodgkin-Huxley m-gate)", fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ablation_mse_vs_size.png"), dpi=200)
        plt.close()
        print(f"Saved: {fig_dir}/ablation_mse_vs_size.png")

        print(f"\n{'N_train':>8}  {'DD MSE':>14}  {'PINN MSE':>14}  {'Improvement':>12}")
        print("-" * 54)
        for i, n in enumerate(sizes):
            improv = (dd_means[i] - pinn_means[i]) / dd_means[i] * 100 if dd_means[i] > 0 else 0
            print(f"{n:>8d}  {dd_means[i]:.6f}+/-{dd_stds[i]:.4f}  {pinn_means[i]:.6f}+/-{pinn_stds[i]:.4f}  {improv:>+10.1f}%")
    else:
        print(f"Ablation results not found at {ablation_path}")

    # ==================== FULL-DATA COMPARISON ====================
    full_path = os.path.join(cfg["paths"]["data_results"], "full_train_results.npz")
    meta_path = os.path.join(cfg["paths"]["data_results"], "full_train_meta.json")

    if os.path.exists(full_path) and os.path.exists(meta_path):
        res = np.load(full_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        dd_pred = res["dd_pred"]
        pinn_pred = res["pinn_pred"]
        train_idx = np.array(meta["train_idx"])
        test_idx = np.array(meta["test_idx"])

        dd_train = compute_metrics(m_true, dd_pred, train_idx)
        dd_test = compute_metrics(m_true, dd_pred, test_idx)
        pinn_train = compute_metrics(m_true, pinn_pred, train_idx)
        pinn_test = compute_metrics(m_true, pinn_pred, test_idx)

        print(f"\n{'':>14}  {'DD Train':>12}  {'DD Test':>12}  {'PINN Train':>12}  {'PINN Test':>12}")
        print("-" * 70)
        for metric in ["MSE", "MAE", "RMSE", "Max Error"]:
            print(f"{metric:>14}  {dd_train[metric]:>12.6f}  {dd_test[metric]:>12.6f}  {pinn_train[metric]:>12.6f}  {pinn_test[metric]:>12.6f}")

        # --- Overlay prediction plot ---
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        axs[0].plot(t, V, "b-", lw=1.5)
        axs[0].set_ylabel("Voltage (mV)", fontsize=12)
        axs[0].set_title("Voltage Step Protocol", fontsize=13)
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(t, m_true, "k-", label="Ground Truth", lw=2.5, alpha=0.8)
        axs[1].plot(t, dd_pred, "--", color="#e74c3c", label="Data-Driven (ReLU)", lw=2)
        axs[1].plot(t, pinn_pred, "--", color="#2ecc71", label="PINN (Tanh + ODE)", lw=2)
        axs[1].set_ylabel("m(t)", fontsize=12)
        axs[1].set_title("Prediction Comparison", fontsize=13)
        axs[1].legend(fontsize=11)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(t, np.abs(m_true - dd_pred), color="#e74c3c", label="Data-Driven |error|", lw=1.5, alpha=0.8)
        axs[2].plot(t, np.abs(m_true - pinn_pred), color="#2ecc71", label="PINN |error|", lw=1.5, alpha=0.8)
        axs[2].set_xlabel("Time (ms)", fontsize=12)
        axs[2].set_ylabel("Absolute Error", fontsize=12)
        axs[2].set_title("Pointwise Error", fontsize=13)
        axs[2].legend(fontsize=11)
        axs[2].grid(True, alpha=0.3)

        plt.suptitle("PINN vs. Data-Driven: Full Comparison on HH m-gate", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "comparison_overlay.png"), dpi=200)
        plt.close()
        print(f"Saved: {fig_dir}/comparison_overlay.png")

        # --- Training dynamics ---
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(res["dd_train_losses"], label="DD Train", color="#e74c3c")
        axs[0].plot(res["dd_test_losses"], label="DD Test", color="#e74c3c", ls="--", alpha=0.7)
        axs[0].plot(res["pinn_train_losses"], label="PINN Total", color="#2ecc71")
        axs[0].plot(res["pinn_test_losses"], label="PINN Test", color="#2ecc71", ls="--", alpha=0.7)
        axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
        axs[0].set_title("Training Curves")
        axs[0].set_yscale("log")
        axs[0].legend(); axs[0].grid(True, alpha=0.3)

        axs[1].plot(res["pinn_data_losses"], label="Data Loss", color="#3498db")
        axs[1].plot(res["pinn_phys_losses"], label="Physics Loss", color="#9b59b6")
        axs[1].plot(res["pinn_train_losses"], label="Total Loss", color="#2ecc71", lw=2)
        axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Loss")
        axs[1].set_title("PINN Loss Decomposition")
        axs[1].set_yscale("log")
        axs[1].legend(); axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "training_dynamics.png"), dpi=200)
        plt.close()
        print(f"Saved: {fig_dir}/training_dynamics.png")
    else:
        print("Full training results not found — run experiments/train_full.py first.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()