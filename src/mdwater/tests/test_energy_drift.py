import numpy as np
from mdwater.sim import MDSim
from mdwater.water import place_waters_grid, water_forces


def test_energy_drift_short_run():
    x0, types, molecules = place_waters_grid(n_side=2, spacing=3.5)
    N = len(x0)
    rng = np.random.default_rng(0)
    v0 = 0.02 * (rng.standard_normal((N, 3)) - 0.5)
    m = np.where(types == "O", 16.0, 1.0)

    params = dict(
        k_bond=300.0,
        r_eq=0.9572,
        k_theta=50.0,
        theta_eq_deg=104.52,
        q_H=+0.4238,
        q_O=-0.8476,
        kC=1.0,
        epsilon_OO=0.2,
        sigma_OO=3.166,
    )

    def force_fn(x, **kw):
        return water_forces(x, types, molecules, params, rcut=6.0, box=None)

    dt = 5e-4
    sim = MDSim(x0, v0, m, force_fn, dt)

    E0 = None
    for _ in range(1000):
        stats = sim.step()
        if E0 is None:
            E0 = stats["E"]

    rel = abs(stats["E"] - E0) / abs(E0)
    assert rel < 1e-3
