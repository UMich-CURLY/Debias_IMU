import torch
from Third_party.torchdiffeq._impl.rk_common import RKAdaptiveStepsizeODESolver
from Third_party.torchdiffeq._impl.solvers import FixedGridODESolver
from Third_party.torchdiffeq._impl.interp import _interp_evaluate
from .lie_util import SO3exp

class SO3FixedGridODESolver(FixedGridODESolver):
    def __init__(self, func, y0, R0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, step_size, grid_constructor, interp, perturb, **unused_kwargs)
        self.R0 = R0.to(y0.dtype).to(y0.device)
    
    # override the function
    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        R_sol = torch.empty(*solution.shape[:-1], 3, 3, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        R_sol[0] = self.R0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            ## check whether the chart needs to be changed ##
            # if y0[..., :3].norm(dim=-1).max() > torch.pi:
            #     self.R0 = self.R0 @ SO3exp(y0[..., :3].clone())
            #     y0[...,:3] = torch.zeros_like(y0[...,:3])
            #     self.func.callback_change_chart(self.R0)
            self.R0 = self.R0 @ SO3exp(y0[..., :3].clone())
            y0[...,:3] = torch.zeros_like(y0[...,:3])
            self.func.callback_change_chart(self.R0)
            #################################################
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy
            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    sol = self._linear_interp(t0, t1, y0, y1, t[j])
                    solution[j] = sol
                    R_sol[j] = self.R0 @ SO3exp(sol[...,:3].clone())
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    sol = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                    solution[j] = sol
                    R_sol[j] = self.R0 @ SO3exp(sol[...,:3].clone())
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution, R_sol

class SO3RKAdaptiveStepsizeODESolver(RKAdaptiveStepsizeODESolver):
    def __init__(self, func, y0, R0, rtol, atol, **options): #(func=func, y0=y0, rtol=rtol, atol=atol, **options)
        super().__init__(func, y0, rtol, atol, **options)
        self.R0 = R0.to(y0.dtype).to(y0.device)
                                                            
    # override the integrate method
    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        R_sol = torch.empty(*solution.shape[:-1], 3, 3, dtype=self.y0.dtype, device=self.y0.device)
        R_sol[0] = self.R0
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i], R_sol[i] = self._advance(t[i])
        return solution, R_sol
    
    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            ## check whether the chart needs to be changed
            if self.rk_state.y1[..., :3].norm(dim=-1).max() > torch.pi:
                self.y0 = self.rk_state.y1.clone()
                self.y0[...,:3] = torch.zeros_like(self.y0[...,:3])
                self.R0 = self.R0 @ SO3exp(self.rk_state.y1[..., :3])
                self.func.callback_change_chart(self.R0)
                self._before_integrate(self.rk_state.t1.unsqueeze(0))
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        sol = _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)
        R_sol = self.R0 @ SO3exp(sol[..., :3])
        return sol, R_sol
