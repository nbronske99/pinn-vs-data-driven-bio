# Physics-Informed vs. Data-Driven Neural Networks for Biological ODE Systems

**BHCC Honors Seminar in AI (HON200) — Spring 2026**

## Research Question

Can embedding known differential equations into a neural network's loss function (the PINN approach) outperform purely data-driven models when learning biological dynamics, particularly under sparse and noisy data conditions?

## System Under Study

The Hodgkin-Huxley sodium activation gating variable *m(t)*, governed by:

dm/dt = α(V)(1 − m) − β(V)m

This ODE describes how voltage-gated ion channels open and close in neurons — a foundational mechanism in bioelectricity.

## Project Structure

- `notebooks/synthetic_data_generator.ipynb` — Generates synthetic m(t) data from the HH equations under a voltage step protocol
- `notebooks/02_train_data_driven_baseline.ipynb` — Trains a standard feedforward neural network on (t, V) → m(t)
- `notebooks/03_train_pinn.ipynb` — Trains a PINN that embeds the ODE residual directly into the loss function
- `data/` — Saved synthetic datasets
- `outputs/` — Result plots

## Status

Work in progress. Upcoming: sparse-data ablation study, inverse parameter recovery, train/test split, and activation function improvements.

## Requirements

Python 3.x, PyTorch, NumPy, SciPy, Matplotlib
