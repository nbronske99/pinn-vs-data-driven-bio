# Physics-Informed vs. Data-Driven Neural Networks for Biological ODE Systems

**BHCC Honors Seminar in AI (HON200) — Spring 2026**

## Research Question

Does embedding the differential equation that governs a biological system into a neural network's loss function improve the model's accuracy compared to a neural network that learns only from data?

## Hypothesis

The model with the embedded equation will be more accurate when training data is limited, but this advantage will shrink as more training data is provided.

## System Under Study

The Hodgkin-Huxley sodium activation gating variable *m(t)*, governed by:

    dm/dt = α(V)(1 − m) − β(V)m

This first-order ODE describes how voltage-gated ion channels open and close in neurons — a fundamental mechanism in bioelectricity.

## How to Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh
```

This generates data, runs all experiments, and produces figures. Results land in `data/results/` and figures in `analysis/figures/`.

## Project Structure
```
configs/experiment.yaml    ← every tunable parameter
src/hh_physics.py          ← Hodgkin-Huxley rate functions (NumPy + PyTorch)
src/models.py              ← DataDrivenNN and PINN architectures
experiments/
  generate_data.py         ← Step 1: solve the ODE, save ground truth
  run_ablation.py          ← Step 2: train both models at 25–800 data points
  train_full.py            ← Step 3: full 80/20 split training with curves
analysis/
  analyze.py               ← Step 4: generate figures and metrics tables
  figures/                 ← output plots
data/raw/                  ← generated synthetic data
data/results/              ← raw experiment outputs (JSON, npz, weights)
docs/                      ← paper, poster
run_all.sh                 ← single command reproduction
```

## Primary Metric

Test MSE (mean squared error) on held-out data points.

## Requirements

Python 3.x, PyTorch, NumPy, SciPy, Matplotlib, PyYAML