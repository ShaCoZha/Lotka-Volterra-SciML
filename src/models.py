from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torchdiffeq import odeint


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class BaselineNN(nn.Module):
    """Plain feed-forward network that maps time directly to state."""

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(in_dim, out_dim, hidden_dim, depth)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


MLP = BaselineNN


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, activation: type[nn.Module] = nn.Tanh) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.Linear(dim, dim),
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layer(x))


class ResNet(nn.Module):
    """Residual network used for direct trajectory regression."""

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*(ResNetBlock(hidden_dim) for _ in range(num_blocks)))
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.in_layer(t))
        x = self.blocks(x)
        return self.out_layer(x)


class NeuralODEFunc(nn.Module):
    """Learned vector field for the state dynamics."""

    def __init__(self, state_dim: int = 2, hidden_dim: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.net = _build_mlp(state_dim, state_dim, hidden_dim, depth)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(state)


class NeuralODEModel(nn.Module):
    """Neural ODE that integrates a learned vector field from an initial state."""

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 2,
        solver: str = "rk4",
        step_size: float | None = None,
    ) -> None:
        super().__init__()
        self.func = NeuralODEFunc(state_dim=state_dim, hidden_dim=hidden_dim, depth=depth)
        self.solver = solver
        self.step_size = step_size
        self.register_buffer("default_initial_state", torch.zeros(state_dim))

    def set_default_initial_state(self, state: torch.Tensor | Iterable[float]) -> None:
        state_tensor = torch.as_tensor(state, dtype=self.default_initial_state.dtype)
        self.default_initial_state.copy_(state_tensor.view(-1))

    def trajectory(
        self,
        t: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_points = t.view(-1)
        if initial_state is None:
            initial_state = self.default_initial_state
        state0 = initial_state.view(-1).to(time_points.device, dtype=time_points.dtype)

        options = {"step_size": self.step_size} if self.solver == "rk4" and self.step_size else None
        trajectory = odeint(
            self.func,
            state0,
            time_points,
            method=self.solver,
            options=options,
            rtol=1e-6,
            atol=1e-7,
        )
        return trajectory

    def forward(
        self,
        t: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trajectory(t, initial_state=initial_state)
