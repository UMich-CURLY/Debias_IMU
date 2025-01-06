from Third_party.torchdiffeq._impl.misc import _check_inputs
from .SO3dopri8 import SO3Dopri8Solver
from .SO3dopri5 import SO3Dopri5Solver
from .SO3bosh3 import SO3Bosh3Solver
from .SO3fehlberg2 import SO3Fehlberg2
from .SO3adaptive_heun import SO3AdaptiveHeunSolver
from .SO3fixed_grid import SO3Euler, SO3Midpoint, SO3Heun3, SO3RK4



SOLVERS = {
    'dopri8': SO3Dopri8Solver,
    'dopri5': SO3Dopri5Solver,
    'bosh3': SO3Bosh3Solver,
    'fehlberg2': SO3Fehlberg2,
    'adaptive_heun': SO3AdaptiveHeunSolver,
    'euler': SO3Euler,
    'midpoint': SO3Midpoint,
    'heun3': SO3Heun3,
    'rk4': SO3RK4,
    ### following is not implemented yet, should be similar to the above ###
    # 'explicit_adams': AdamsBashforth,
    # 'implicit_adams': AdamsBashforthMoulton,
    # # Backward compatibility: use the same name as before
    # 'fixed_adams': AdamsBashforthMoulton,
    # # ~Backwards compatibility
    # 'scipy_solver': ScipyWrapperODESolver,
}

def odeint_SO3(func, y0, R0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """method is 'dopri5' by default"""
    
    original_func = func
    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)
    ### add callback function to change the chart
    try:
        callback = getattr(original_func, 'callback_change_chart')
    except AttributeError:
        callback = lambda x: None
    else:
        pass
    setattr(func, 'callback_change_chart', callback)

    solver = SOLVERS[method](func=func, y0=y0, R0=R0, rtol=rtol, atol=atol, **options) 

    solution, R_sol = solver.integrate(t)
    return solution, R_sol


## test ##