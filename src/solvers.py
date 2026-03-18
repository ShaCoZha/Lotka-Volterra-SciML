from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np


ArrayLike = np.ndarray | Iterable[float]


@dataclass(frozen=True)
class LotkaVolterraParams:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0


def lotka_volterra(
    state: ArrayLike,
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> np.ndarray:
    state_array = np.asarray(state, dtype=float)
    x = state_array[..., 0]
    y = state_array[..., 1]
    dx = params.alpha * x - params.beta * x * y
    dy = params.delta * x * y - params.gamma * y
    return np.stack((dx, dy), axis=-1)


def jacobian(
    state: ArrayLike,
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> np.ndarray:
    x, y = np.asarray(state, dtype=float)
    return np.array(
        [
            [params.alpha - params.beta * y, -params.beta * x],
            [params.delta * y, params.delta * x - params.gamma],
        ],
        dtype=float,
    )


def equilibria(params: LotkaVolterraParams = LotkaVolterraParams()) -> dict[str, np.ndarray]:
    return {
        "origin": np.array([0.0, 0.0], dtype=float),
        "coexistence": np.array([params.gamma / params.delta, params.alpha / params.beta], dtype=float),
    }


def nullclines(params: LotkaVolterraParams = LotkaVolterraParams()) -> dict[str, float]:
    return {
        "dx_dt_zero_y": params.alpha / params.beta,
        "dy_dt_zero_x": params.gamma / params.delta,
    }


def coexistence_eigenvalues(params: LotkaVolterraParams = LotkaVolterraParams()) -> np.ndarray:
    return np.linalg.eigvals(jacobian(equilibria(params)["coexistence"], params))


def small_oscillation_period(params: LotkaVolterraParams = LotkaVolterraParams()) -> float:
    return 2.0 * np.pi / np.sqrt(params.alpha * params.gamma)


def conserved_quantity(
    state: ArrayLike,
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> np.ndarray:
    state_array = np.asarray(state, dtype=float)
    x = state_array[..., 0]
    y = state_array[..., 1]
    valid = (x > 0.0) & (y > 0.0)

    values = np.full_like(x, np.nan, dtype=float)
    values[valid] = (
        params.delta * x[valid]
        - params.gamma * np.log(x[valid])
        + params.beta * y[valid]
        - params.alpha * np.log(y[valid])
    )
    return values


def solve_numerical(
    method: str = "rk4",
    dt: float = 0.05,
    t_max: float = 20.0,
    init_state: ArrayLike = (2.0, 1.0),
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n_steps = int(round(t_max / dt))
    t_steps = np.linspace(0.0, t_max, n_steps + 1, dtype=float)
    h = t_steps[1] - t_steps[0]
    states = np.zeros((len(t_steps), 2), dtype=float)
    states[0] = np.asarray(init_state, dtype=float)

    for i in range(len(t_steps) - 1):
        current = states[i]
        k1 = lotka_volterra(current, params)

        if method == "euler":
            states[i + 1] = current + h * k1
        elif method == "rk2":
            k2 = lotka_volterra(current + 0.5 * h * k1, params)
            states[i + 1] = current + h * k2
        elif method == "rk4":
            k2 = lotka_volterra(current + 0.5 * h * k1, params)
            k3 = lotka_volterra(current + 0.5 * h * k2, params)
            k4 = lotka_volterra(current + h * k3, params)
            states[i + 1] = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown method '{method}'. Expected one of: euler, rk2, rk4.")

    return t_steps, states


def reference_solution(
    t_max: float = 20.0,
    dt: float = 1e-3,
    init_state: ArrayLike = (2.0, 1.0),
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> tuple[np.ndarray, np.ndarray]:
    return solve_numerical(method="rk4", dt=dt, t_max=t_max, init_state=init_state, params=params)


def interpolate_reference(
    target_t: np.ndarray,
    ref_t: np.ndarray,
    ref_states: np.ndarray,
) -> np.ndarray:
    x_interp = np.interp(target_t, ref_t, ref_states[:, 0])
    y_interp = np.interp(target_t, ref_t, ref_states[:, 1])
    return np.column_stack((x_interp, y_interp))


def solver_metrics(
    t_steps: np.ndarray,
    states: np.ndarray,
    ref_t: np.ndarray,
    ref_states: np.ndarray,
    params: LotkaVolterraParams = LotkaVolterraParams(),
) -> dict[str, float]:
    ref_interp = interpolate_reference(t_steps, ref_t, ref_states)
    errors = np.linalg.norm(states - ref_interp, axis=1)
    invariant = conserved_quantity(states, params)
    valid_invariant = invariant[np.isfinite(invariant)]
    invariant_drift = float(np.max(np.abs(valid_invariant - valid_invariant[0]))) if valid_invariant.size else np.inf

    return {
        "mean_state_error": float(np.mean(errors)),
        "max_state_error": float(np.max(errors)),
        "final_state_error": float(errors[-1]),
        "positivity_violations": float(np.sum(np.any(states <= 0.0, axis=1))),
        "max_invariant_drift": invariant_drift,
    }


def benchmark_solvers(
    methods: Iterable[str],
    dts: Iterable[float],
    t_max: float = 20.0,
    init_state: ArrayLike = (2.0, 1.0),
    params: LotkaVolterraParams = LotkaVolterraParams(),
    reference_dt: float = 1e-3,
) -> tuple[list[dict[str, float | str]], dict[tuple[str, float], tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    ref_t, ref_states = reference_solution(t_max=t_max, dt=reference_dt, init_state=init_state, params=params)

    rows: list[dict[str, float | str]] = []
    trajectories: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    for method in methods:
        for dt in dts:
            start = perf_counter()
            t_steps, states = solve_numerical(
                method=method,
                dt=dt,
                t_max=t_max,
                init_state=init_state,
                params=params,
            )
            runtime = perf_counter() - start
            metrics = solver_metrics(t_steps, states, ref_t, ref_states, params=params)
            rows.append(
                {
                    "method": method,
                    "dt": float(dt),
                    "runtime_seconds": float(runtime),
                    **metrics,
                }
            )
            trajectories[(method, float(dt))] = (t_steps, states)

    return rows, trajectories, (ref_t, ref_states)
