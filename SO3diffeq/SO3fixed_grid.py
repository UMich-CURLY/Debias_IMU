from .SO3solver import SO3FixedGridODESolver
from Third_party.torchdiffeq._impl.rk_common import rk4_alt_step_func, rk3_step_func
from Third_party.torchdiffeq._impl.misc import Perturb


class SO3Euler(SO3FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


class SO3Midpoint(SO3FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0


class SO3RK4(SO3FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0


class SO3Heun3(SO3FixedGridODESolver):
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)

        butcher_tableu = [
            [0.0, 0.0, 0.0, 0.0],
            [1/3, 1/3, 0.0, 0.0],
            [2/3, 0.0, 2/3, 0.0],
            [0.0, 1/4, 0.0, 3/4],
        ]

        return rk3_step_func(func, t0, dt, t1, y0, butcher_tableu=butcher_tableu, f0=f0, perturb=self.perturb), f0