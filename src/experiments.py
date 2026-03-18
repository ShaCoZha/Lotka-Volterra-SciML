from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .models import BaselineNN, NeuralODEModel, ResNet
from .solvers import (
    LotkaVolterraParams,
    benchmark_solvers,
    conserved_quantity,
    coexistence_eigenvalues,
    equilibria,
    interpolate_reference,
    nullclines,
    reference_solution,
    small_oscillation_period,
)
from .train import (
    NeuralODETrainingConfig,
    SupervisedTrainingConfig,
    predict_trajectory,
    set_seed,
    train_neural_ode_model,
    train_supervised_model,
    trajectory_metrics,
)


@dataclass
class SolverStudyConfig:
    methods: tuple[str, ...] = ("euler", "rk2", "rk4")
    dts: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025)
    t_max: float = 20.0
    reference_dt: float = 1e-3
    init_state: tuple[float, float] = (2.0, 1.0)
    params: LotkaVolterraParams = field(default_factory=LotkaVolterraParams)


@dataclass
class MLStudyConfig:
    train_horizon: float = 8.0
    eval_horizon: float = 20.0
    eval_dt: float = 0.05
    reference_dt: float = 1e-3
    observation_stride: int = 4
    init_state: tuple[float, float] = (2.0, 1.0)
    params: LotkaVolterraParams = field(default_factory=LotkaVolterraParams)
    seed: int = 42
    baseline_train: SupervisedTrainingConfig = field(
        default_factory=lambda: SupervisedTrainingConfig(epochs=1800, lr=1e-3, verbose_every=300)
    )
    resnet_train: SupervisedTrainingConfig = field(
        default_factory=lambda: SupervisedTrainingConfig(epochs=1800, lr=8e-4, verbose_every=300)
    )
    pinn_train: SupervisedTrainingConfig = field(
        default_factory=lambda: SupervisedTrainingConfig(
            epochs=2200,
            lr=1e-3,
            verbose_every=400,
            physics_weight=5e-2,
            initial_weight=10.0,
            collocation_factor=6,
        )
    )
    neural_ode_train: NeuralODETrainingConfig = field(
        default_factory=lambda: NeuralODETrainingConfig(epochs=900, lr=3e-3, verbose_every=150)
    )


def system_summary(params: LotkaVolterraParams = LotkaVolterraParams()) -> dict[str, Any]:
    eq = equilibria(params)
    return {
        "equilibria": {key: value.tolist() for key, value in eq.items()},
        "coexistence_eigenvalues": [complex(ev).real if abs(complex(ev).imag) < 1e-12 else str(complex(ev)) for ev in coexistence_eigenvalues(params)],
        "nullclines": nullclines(params),
        "small_oscillation_period": float(small_oscillation_period(params)),
    }


def format_metrics_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _max_invariant_drift(states: np.ndarray, params: LotkaVolterraParams) -> float:
    invariant = conserved_quantity(states, params)
    valid = invariant[np.isfinite(invariant)]
    if valid.size == 0:
        return float("inf")
    return float(np.max(np.abs(valid - valid[0])))


def _plot_reference_trajectory(
    ref_t: np.ndarray,
    ref_states: np.ndarray,
    params: LotkaVolterraParams,
    path: Path,
) -> None:
    eq = equilibria(params)["coexistence"]
    nc = nullclines(params)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ref_t, ref_states[:, 0], label="Prey x(t)", linewidth=2.0)
    axes[0].plot(ref_t, ref_states[:, 1], label="Predator y(t)", linewidth=2.0)
    axes[0].axhline(eq[0], color="tab:blue", linestyle="--", alpha=0.5)
    axes[0].axhline(eq[1], color="tab:orange", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].set_title("Reference Trajectory")
    axes[0].legend()

    axes[1].plot(ref_states[:, 0], ref_states[:, 1], color="black", linewidth=2.0)
    axes[1].axvline(nc["dy_dt_zero_x"], color="tab:green", linestyle="--", label="x = gamma / delta")
    axes[1].axhline(nc["dx_dt_zero_y"], color="tab:red", linestyle="--", label="y = alpha / beta")
    axes[1].scatter([eq[0]], [eq[1]], color="tab:purple", s=50, label="Coexistence equilibrium")
    axes[1].set_xlabel("Prey x")
    axes[1].set_ylabel("Predator y")
    axes[1].set_title("Phase Portrait")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_solver_diagnostics(
    solver_rows: list[dict[str, Any]],
    trajectories: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]],
    reference: tuple[np.ndarray, np.ndarray],
    params: LotkaVolterraParams,
    output_dir: Path,
    dt_to_show: float = 0.1,
) -> None:
    ref_t, ref_states = reference
    methods = sorted({str(row["method"]) for row in solver_rows})
    colors = {"euler": "tab:red", "rk2": "tab:green", "rk4": "tab:blue"}

    fig, axes = plt.subplots(1, len(methods), figsize=(4.8 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        t_steps, states = trajectories[(method, dt_to_show)]
        ax.plot(ref_states[:, 0], ref_states[:, 1], color="black", linewidth=2.0, label="Reference RK4")
        ax.plot(states[:, 0], states[:, 1], color=colors[method], linewidth=1.8, label=f"{method} dt={dt_to_show}")
        ax.set_title(f"{method.upper()} Phase Portrait")
        ax.set_xlabel("Prey x")
        ax.set_ylabel("Predator y")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "solver_phase_portraits.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in methods:
        subset = sorted((row for row in solver_rows if row["method"] == method), key=lambda row: float(row["dt"]))
        dts = np.array([row["dt"] for row in subset], dtype=float)
        max_errors = np.array([row["max_state_error"] for row in subset], dtype=float)
        invariant_drift = np.array([row["max_invariant_drift"] for row in subset], dtype=float)
        axes[0].loglog(dts, max_errors, marker="o", linewidth=2.0, label=method.upper())
        axes[1].loglog(dts, invariant_drift, marker="o", linewidth=2.0, label=method.upper())

    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    axes[0].set_title("Global State Error vs Step Size")
    axes[1].set_title("Invariant Drift vs Step Size")
    axes[0].set_xlabel("dt")
    axes[1].set_xlabel("dt")
    axes[0].set_ylabel("max ||u - u_ref||_2")
    axes[1].set_ylabel("max |H(t) - H(0)|")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "solver_error_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_learning_problem(config: MLStudyConfig) -> dict[str, Any]:
    ref_t, ref_states = reference_solution(
        t_max=config.eval_horizon,
        dt=config.reference_dt,
        init_state=config.init_state,
        params=config.params,
    )
    eval_t = np.arange(0.0, config.eval_horizon + 0.5 * config.eval_dt, config.eval_dt)
    eval_states = interpolate_reference(eval_t, ref_t, ref_states)

    train_mask = eval_t <= config.train_horizon + 1e-12
    train_t_dense = eval_t[train_mask]
    train_states_dense = eval_states[train_mask]

    observation_idx = np.arange(0, len(train_t_dense), config.observation_stride)
    t_train = train_t_dense[observation_idx][:, None]
    y_train = train_states_dense[observation_idx]
    split_index = int(np.searchsorted(eval_t, config.train_horizon, side="right"))

    return {
        "t_eval": eval_t[:, None],
        "y_eval": eval_states,
        "t_train": t_train,
        "y_train": y_train,
        "train_initial_state": y_train[0],
        "split_index": split_index,
        "reference_t": ref_t,
        "reference_states": ref_states,
    }


def _plot_ml_predictions(
    problem: dict[str, Any],
    predictions: dict[str, np.ndarray],
    histories: dict[str, list[dict[str, float]]],
    params: LotkaVolterraParams,
    output_dir: Path,
) -> None:
    t_eval = problem["t_eval"].reshape(-1)
    y_eval = problem["y_eval"]
    t_train = problem["t_train"].reshape(-1)
    y_train = problem["y_train"]
    train_horizon = t_train[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t_eval, y_eval[:, 0], color="black", linewidth=2.0, label="Reference")
    axes[0, 1].plot(t_eval, y_eval[:, 1], color="black", linewidth=2.0, label="Reference")
    axes[0, 0].scatter(t_train, y_train[:, 0], color="black", s=18, marker="x", label="Observed train points")
    axes[0, 1].scatter(t_train, y_train[:, 1], color="black", s=18, marker="x", label="Observed train points")

    for name, pred in predictions.items():
        axes[0, 0].plot(t_eval, pred[:, 0], linewidth=1.5, label=name)
        axes[0, 1].plot(t_eval, pred[:, 1], linewidth=1.5, label=name)
        axes[1, 0].plot(pred[:, 0], pred[:, 1], linewidth=1.6, label=name)

        invariant = conserved_quantity(pred, params)
        valid = np.isfinite(invariant)
        if np.any(valid):
            axes[1, 1].plot(
                t_eval[valid],
                np.abs(invariant[valid] - invariant[valid][0]),
                linewidth=1.6,
                label=name,
            )

    axes[0, 0].axvline(train_horizon, color="gray", linestyle="--")
    axes[0, 1].axvline(train_horizon, color="gray", linestyle="--")
    axes[0, 0].set_title("Prey Prediction")
    axes[0, 1].set_title("Predator Prediction")
    axes[1, 0].set_title("Predicted Phase Portraits")
    axes[1, 1].set_title("Invariant Drift of Learned Trajectories")
    axes[0, 0].set_ylabel("x(t)")
    axes[0, 1].set_ylabel("y(t)")
    axes[1, 0].set_xlabel("Prey x")
    axes[1, 0].set_ylabel("Predator y")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("|H(t) - H(0)|")
    for ax in axes.flat:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "ml_prediction_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, history in histories.items():
        epochs = [entry["epoch"] for entry in history]
        losses = [entry["total"] for entry in history]
        ax.plot(epochs, losses, linewidth=1.7, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training Curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "ml_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_solver_study(
    config: SolverStudyConfig | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or SolverStudyConfig()
    rows, trajectories, reference = benchmark_solvers(
        methods=config.methods,
        dts=config.dts,
        t_max=config.t_max,
        init_state=config.init_state,
        params=config.params,
        reference_dt=config.reference_dt,
    )

    summary = {
        "config": asdict(config),
        "system_summary": system_summary(config.params),
        "metrics": rows,
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        _ensure_dir(output_path)
        _write_csv(output_path / "solver_metrics.csv", rows)
        _write_json(output_path / "solver_summary.json", summary)
        _plot_reference_trajectory(reference[0], reference[1], config.params, output_path / "reference_trajectory.png")
        _plot_solver_diagnostics(rows, trajectories, reference, config.params, output_path)

    return {
        "rows": rows,
        "trajectories": trajectories,
        "reference": reference,
        "summary": summary,
    }


def run_ml_study(
    config: MLStudyConfig | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or MLStudyConfig()
    set_seed(config.seed)
    problem = build_learning_problem(config)

    baseline = BaselineNN(hidden_dim=64, depth=3)
    resnet = ResNet(hidden_dim=64, num_blocks=4)
    pinn = BaselineNN(hidden_dim=64, depth=3)
    neural_ode = NeuralODEModel(hidden_dim=48, depth=2, solver="rk4", step_size=config.eval_dt / 2.0)

    _, baseline_history = train_supervised_model(
        baseline,
        problem["t_train"],
        problem["y_train"],
        params=config.params,
        config=config.baseline_train,
    )
    _, resnet_history = train_supervised_model(
        resnet,
        problem["t_train"],
        problem["y_train"],
        params=config.params,
        config=config.resnet_train,
    )
    _, pinn_history = train_supervised_model(
        pinn,
        problem["t_train"],
        problem["y_train"],
        params=config.params,
        config=config.pinn_train,
        is_pinn=True,
        collocation_horizon=config.eval_horizon,
    )
    _, neural_ode_history = train_neural_ode_model(
        neural_ode,
        problem["t_train"],
        problem["y_train"],
        config=config.neural_ode_train,
    )

    predictions = {
        "Baseline NN": predict_trajectory(baseline, problem["t_eval"]),
        "ResNet": predict_trajectory(resnet, problem["t_eval"]),
        "PINN": predict_trajectory(pinn, problem["t_eval"]),
        "Neural ODE": predict_trajectory(
            neural_ode,
            problem["t_eval"].reshape(-1),
            initial_state=problem["train_initial_state"],
        ),
    }
    histories = {
        "Baseline NN": baseline_history,
        "ResNet": resnet_history,
        "PINN": pinn_history,
        "Neural ODE": neural_ode_history,
    }

    rows: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        metrics = trajectory_metrics(pred, problem["y_eval"], problem["split_index"])
        metrics["max_invariant_drift"] = _max_invariant_drift(pred, config.params)
        rows.append({"model": name, **metrics})

    summary = {
        "config": asdict(config),
        "metrics": rows,
        "system_summary": system_summary(config.params),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        _ensure_dir(output_path)
        _write_csv(output_path / "ml_metrics.csv", rows)
        _write_json(output_path / "ml_summary.json", summary)
        _plot_ml_predictions(problem, predictions, histories, config.params, output_path)

    return {
        "problem": problem,
        "predictions": predictions,
        "histories": histories,
        "rows": rows,
        "summary": summary,
    }


def run_full_analysis(
    *,
    output_dir: str | Path = "results",
    quick: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    _ensure_dir(output_path)

    solver_config = SolverStudyConfig()
    ml_config = MLStudyConfig()

    if quick:
        solver_config = SolverStudyConfig(dts=(0.2, 0.1, 0.05), t_max=15.0)
        ml_config = MLStudyConfig(
            train_horizon=6.0,
            eval_horizon=15.0,
            baseline_train=SupervisedTrainingConfig(epochs=400, lr=1e-3, verbose_every=100),
            resnet_train=SupervisedTrainingConfig(epochs=400, lr=8e-4, verbose_every=100),
            pinn_train=SupervisedTrainingConfig(
                epochs=500,
                lr=1e-3,
                verbose_every=100,
                physics_weight=5e-2,
                initial_weight=10.0,
                collocation_factor=4,
            ),
            neural_ode_train=NeuralODETrainingConfig(epochs=250, lr=3e-3, verbose_every=50),
        )

    solver_results = run_solver_study(solver_config, output_dir=output_path)
    ml_results = run_ml_study(ml_config, output_dir=output_path)

    overview = {
        "solver_metrics": solver_results["rows"],
        "ml_metrics": ml_results["rows"],
        "solver_table": format_metrics_table(
            solver_results["rows"],
            ["method", "dt", "max_state_error", "final_state_error", "max_invariant_drift", "runtime_seconds"],
        ),
        "ml_table": format_metrics_table(
            ml_results["rows"],
            ["model", "train_rmse", "extrap_rmse", "final_state_error", "max_invariant_drift"],
        ),
    }
    _write_json(output_path / "overview.json", overview)

    summary_lines = [
        "# Experiment Summary",
        "",
        "## Solver Metrics",
        overview["solver_table"],
        "",
        "## ML Metrics",
        overview["ml_table"],
        "",
    ]
    (output_path / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "solver": solver_results,
        "ml": ml_results,
        "overview": overview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Lotka-Volterra analysis pipeline.")
    parser.add_argument("--output-dir", default="results", help="Directory where tables and figures are saved.")
    parser.add_argument("--quick", action="store_true", help="Run a faster but lower-fidelity version of the study.")
    args = parser.parse_args()

    results = run_full_analysis(output_dir=args.output_dir, quick=args.quick)
    print("Saved outputs to:", Path(args.output_dir).resolve())
    print()
    print(results["overview"]["solver_table"])
    print()
    print(results["overview"]["ml_table"])


if __name__ == "__main__":
    main()
