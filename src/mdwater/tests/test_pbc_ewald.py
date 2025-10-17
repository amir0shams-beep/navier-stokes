import numpy as np
from mdwater.sim import MDSim
from mdwater.water import place_waters_grid, water_forces


def test_energy_drift_pbc_ewald():
    x0, types, molecules = place_waters_grid(n_side=2, spacing=3.5)
    N = len(x0)
    rng = np.random.default_rng(0)
    v0 = 0.02 * (rng.standard_normal((N, 3)) - 0.5)
    m = np.where(types == "O", 16.0, 1.0)

    L = 14.0
    params = dict(
        k_bond=300.0,
        r_eq=0.9572,
        k_theta=50.0,
        theta_eq_deg=104.52,
        q_H=+0.4238,
        q_O=-0.8476,
        epsilon_OO=0.2,
        sigma_OO=3.166,
        coulomb_method="ewald",
        ewald_alpha=0.25,
        ewald_r_real=6.0,
        ewald_kmax=6,
    )

    def force_fn(x, **kw):
        return water_forces(
            x, types, molecules, params, box=L, rcut_lj=6.0, rcut_coulomb=None
        )

    sim = MDSim(x0, v0, m, force_fn, dt=5e-4)
    E0 = None
    for _ in range(1500):
        stats = sim.step()
        if E0 is None:
            E0 = stats["E"]
    drift = abs(stats["E"] - E0) / abs(E0)
    assert drift < 5e-4  # conservative bound for CI stability
