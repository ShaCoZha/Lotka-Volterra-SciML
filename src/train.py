from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .solvers import LotkaVolterraParams


@dataclass
class SupervisedTrainingConfig:
    epochs: int = 2000
    lr: float = 1e-3
    weight_decay: float = 0.0
    verbose_every: int = 250
    physics_weight: float = 1e-2
    initial_weight: float = 5.0
    collocation_factor: int = 4


@dataclass
class NeuralODETrainingConfig:
    epochs: int = 1200
    lr: float = 5e-3
    weight_decay: float = 1e-5
    verbose_every: int = 100
    grad_clip: float = 1.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.detach().clone().float()
    return torch.as_tensor(array, dtype=torch.float32)


def _compute_pinn_loss(
    model: nn.Module,
    t_data: torch.Tensor,
    target_data: torch.Tensor,
    t_collocation: torch.Tensor,
    params: LotkaVolterraParams,
    config: SupervisedTrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_data = model(t_data)
    loss_data = torch.mean((pred_data - target_data) ** 2)

    t_collocation = t_collocation.detach().clone().requires_grad_(True)
    pred_phys = model(t_collocation)
    x = pred_phys[:, 0:1]
    y = pred_phys[:, 1:2]

    dx_dt = torch.autograd.grad(x, t_collocation, torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_collocation, torch.ones_like(y), create_graph=True)[0]

    res_x = dx_dt - (params.alpha * x - params.beta * x * y)
    res_y = dy_dt - (params.delta * x * y - params.gamma * y)
    loss_phys = torch.mean(res_x**2) + torch.mean(res_y**2)

    init_pred = model(t_data[:1])
    init_loss = torch.mean((init_pred - target_data[:1]) ** 2)

    total_loss = loss_data + config.physics_weight * loss_phys + config.initial_weight * init_loss
    parts = {
        "data": float(loss_data.item()),
        "physics": float(loss_phys.item()),
        "initial": float(init_loss.item()),
        "total": float(total_loss.item()),
    }
    return total_loss, parts


def train_supervised_model(
    model: nn.Module,
    t_data: np.ndarray | torch.Tensor,
    target_data: np.ndarray | torch.Tensor,
    *,
    params: LotkaVolterraParams = LotkaVolterraParams(),
    config: SupervisedTrainingConfig | None = None,
    is_pinn: bool = False,
    collocation_horizon: float | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    config = config or SupervisedTrainingConfig()
    t_data = to_tensor(t_data)
    target_data = to_tensor(target_data)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    if collocation_horizon is None:
        collocation_horizon = float(t_data.max().item())
    num_collocation = max(len(t_data) * config.collocation_factor, 200)
    t_collocation = torch.linspace(0.0, collocation_horizon, num_collocation).view(-1, 1)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        if is_pinn:
            loss, parts = _compute_pinn_loss(
                model,
                t_data=t_data,
                target_data=target_data,
                t_collocation=t_collocation,
                params=params,
                config=config,
            )
        else:
            pred = model(t_data)
            loss = torch.mean((pred - target_data) ** 2)
            parts = {"data": float(loss.item()), "total": float(loss.item())}

        loss.backward()
        optimizer.step()

        parts["epoch"] = float(epoch)
        history.append(parts)
        if config.verbose_every and epoch % config.verbose_every == 0:
            print(f"Epoch {epoch:04d} | loss={parts['total']:.6f}")

    return model, history


def train_neural_ode_model(
    model: nn.Module,
    t_data: np.ndarray | torch.Tensor,
    target_data: np.ndarray | torch.Tensor,
    *,
    config: NeuralODETrainingConfig | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    config = config or NeuralODETrainingConfig()
    t_data = to_tensor(t_data).view(-1)
    target_data = to_tensor(target_data)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    initial_state = target_data[0]
    if hasattr(model, "set_default_initial_state"):
        model.set_default_initial_state(initial_state)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        pred = model(t_data, initial_state=initial_state)
        loss = torch.mean((pred - target_data) ** 2)
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        entry = {"epoch": float(epoch), "total": float(loss.item())}
        history.append(entry)
        if config.verbose_every and epoch % config.verbose_every == 0:
            print(f"Epoch {epoch:04d} | loss={entry['total']:.6f}")

    return model, history


def predict_trajectory(
    model: nn.Module,
    t_eval: np.ndarray | torch.Tensor,
    *,
    initial_state: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    t_tensor = to_tensor(t_eval)
    with torch.no_grad():
        if initial_state is not None:
            pred = model(t_tensor.view(-1), initial_state=to_tensor(initial_state))
        else:
            pred = model(t_tensor)
    return pred.detach().cpu().numpy()


def trajectory_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    split_index: int,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    errors = np.linalg.norm(pred - target, axis=1)

    return {
        "train_rmse": float(np.sqrt(np.mean(errors[:split_index] ** 2))),
        "extrap_rmse": float(np.sqrt(np.mean(errors[split_index:] ** 2))),
        "final_state_error": float(errors[-1]),
        "max_state_error": float(np.max(errors)),
    }
