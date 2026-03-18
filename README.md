# Comparative Analysis of Lotka-Volterra Dynamics

This repository studies the Lotka-Volterra predator-prey system from four complementary angles:

1. Mathematical and qualitative analysis of equilibria, nullclines, and conserved structure.
2. Numerical analysis of explicit solvers (Euler, RK2, RK4), including stability and invariant drift.
3. Machine learning trajectory modeling with a Baseline NN, ResNet, PINN, and Neural ODE.
4. Reproducible reporting through a notebook plus a script that regenerates all figures and tables.

## Problem Statement

We consider the nonlinear ODE system

\[
\frac{dx}{dt} = \alpha x - \beta x y, \qquad
\frac{dy}{dt} = \delta x y - \gamma y
\]

with positive parameters \(\alpha,\beta,\gamma,\delta\). For the default configuration in this project,
\[
\alpha = \beta = \gamma = \delta = 1,
\]
so the coexistence equilibrium is \((x^\*, y^\*) = (1, 1)\).

### Main mathematical facts used in the project

- Equilibria: \((0,0)\) and \((\gamma/\delta, \alpha/\beta)\).
- Jacobian:
  \[
  J(x,y)=
  \begin{bmatrix}
  \alpha-\beta y & -\beta x \\
  \delta y & \delta x - \gamma
  \end{bmatrix}.
  \]
- At coexistence, the eigenvalues are purely imaginary \(\lambda = \pm i\sqrt{\alpha\gamma}\), so the linearized system behaves like a center.
- The system has the conserved quantity
  \[
  H(x,y)=\delta x - \gamma \ln x + \beta y - \alpha \ln y,
  \]
  which provides a natural diagnostic for numerical and learned trajectories.

## Repository Structure

```text
.
├── main_analysis.ipynb      # Narrative notebook with math, solver, and ML analysis
├── run_analysis.py          # Convenience entry point for the full experiment pipeline
├── requirements.txt
├── README.md
└── src
    ├── __init__.py
    ├── experiments.py       # Reproducible solver/ML studies and plotting
    ├── models.py            # Baseline NN, ResNet, Neural ODE
    ├── solvers.py           # Dynamics, equilibria, invariants, explicit solvers
    └── train.py             # Training loops and evaluation metrics
```

## Reproducibility

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full experiment pipeline

```bash
python run_analysis.py
```

This creates a `results/` directory with:

- `reference_trajectory.png`
- `solver_phase_portraits.png`
- `solver_error_curves.png`
- `ml_prediction_comparison.png`
- `ml_training_curves.png`
- `solver_metrics.csv`
- `ml_metrics.csv`
- `SUMMARY.md`

### 3. Open the notebook

```bash
jupyter notebook main_analysis.ipynb
```

The notebook uses the same code as the script, so the figures and tables are consistent across both workflows.

## What Each Model Represents

- `Baseline NN`: direct regression from time \(t\) to state \((x(t), y(t))\).
- `ResNet`: direct regression with residual blocks to ease deeper optimization.
- `PINN`: the same direct regression architecture, but trained with ODE residual penalties and initial-condition anchoring.
- `Neural ODE`: learns a vector field in state space and integrates it from the initial condition.

## Notes

- All training uses fixed seeds for repeatability.
- The default settings are chosen to keep the project tractable on CPU.
- The notebook is intentionally written as a report, while `src/experiments.py` is the reproducible experiment engine.
