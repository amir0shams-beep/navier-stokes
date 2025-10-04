import numpy as np
from .integrators import velocity_verlet


class MDSim:
    def __init__(self, x, v, m, force_fn, dt, force_kwargs=None):
        self.x = x.astype(float)
        self.v = v.astype(float)
        self.m = m.astype(float)
        self.dt = float(dt)
        self.force_fn = force_fn
        self.kw = {} if force_kwargs is None else dict(force_kwargs)

    def step(self):
        self.x, self.v, f, U = velocity_verlet(
            self.x, self.v, self.m, self.force_fn, self.dt, **self.kw
        )
        K = 0.5 * np.sum(self.m[:, None] * self.v * self.v)
        return {"K": K, "U": U, "E": K + U}
