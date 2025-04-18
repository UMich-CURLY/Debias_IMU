import torch
from Third_party.torchdiffeq._impl.rk_common import _ButcherTableau
from .SO3solver import SO3RKAdaptiveStepsizeODESolver

_FEHLBERG2_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 1.0], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),
        torch.tensor([1 / 256, 255 / 256], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
    c_error=torch.tensor(
        [-1 / 512, 0, 1 / 512], dtype=torch.float64
    ),
)

_FE_C_MID = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)


class SO3Fehlberg2(SO3RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _FEHLBERG2_TABLEAU
    mid = _FE_C_MID