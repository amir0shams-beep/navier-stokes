import numpy as np


def velocity_verlet(x, v, m, force_fn, dt, **force_kwargs):
    f, pot = force_fn(x, **force_kwargs)
    v_half = v + 0.5 * dt * (f / m[:, None])
    x_new = x + dt * v_half
    f_new, pot_new = force_fn(x_new, **force_kwargs)
    v_new = v_half + 0.5 * dt * (f_new / m[:, None])
    return x_new, v_new, f_new, pot_new
