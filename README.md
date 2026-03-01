# Physics-Informed vs. Data-Driven Neural Networks for Biological ODE Systems

**BHCC Honors Seminar in AI (HON200) — Spring 2026**

## Research Question

Can embedding known differential equations into a neural network's loss function (the PINN approach) outperform purely data-driven models when learning biological dynamics, particularly under sparse and noisy data conditions?

## System Under Study

The Hodgkin-Huxley sodium activation gating variable *m(t)*, governed by:

dm/dt = α(V)(1 − m) − β(V)m

This ODE describes how voltage-gated ion channels open and close in neurons. This a key mechanism in bioelectricity, which is a field of interest of mine.

## Project Structure

- `notebooks/synthetic_data_generator.ipynb` — Generates synthetic m(t) data from the HH equations under a voltage step protocol
- `notebooks/02_train_data_driven_baseline.ipynb` — Trains a standard feedforward neural network (ReLU) on (t, V) → m(t) with 80/20 train/test split
- `notebooks/03_train_pinn.ipynb` — Trains a PINN (Tanh activations) that embeds the ODE residual directly into the loss function
- `notebooks/04_sparse_ablation.ipynb` — Core experiment: trains both models on 25–800 data points and measures generalization gap
- `notebooks/05_comparison.ipynb` — Side-by-side metrics, overlay plots, and ablation summary
- `data/` — Saved synthetic datasets and train/test split indices
- `outputs/` — Result plots, saved model weights, and metrics

## Status

Core experiments implemented. Possible extensions: inverse parameter recovery, noise robustness testing, λ_phys hyperparameter sweep.

## Requirements

Python 3.x, PyTorch, NumPy, SciPy, Matplotlib
