import torch
from Third_party.torchdiffeq._impl.rk_common import _ButcherTableau
from .SO3solver import SO3RKAdaptiveStepsizeODESolver


_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=torch.float64),
    beta=[
        torch.tensor([1.], dtype=torch.float64),
    ],
    c_sol=torch.tensor([0.5, 0.5], dtype=torch.float64),
    c_error=torch.tensor([
        0.5,
        -0.5,
    ], dtype=torch.float64),
)

_AH_C_MID = torch.tensor([
    0.5, 0.
], dtype=torch.float64)


class SO3AdaptiveHeunSolver(SO3RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
