import numpy as np
from mdwater.sim import MDSim
from mdwater.water import place_waters_grid, water_forces
from mdwater.pbc import wrap_positions
from mdwater.nlist import NeighborList  # <-- add


def main():
    x0, types, molecules = place_waters_grid(n_side=2, spacing=3.5)
    N = len(x0)
    m = np.where(types == "O", 16.0, 1.0)
    rng = np.random.default_rng(0)
    v0 = 0.02 * (rng.standard_normal((N, 3)) - 0.5)

    L = 14.0
    box = L

    # neighbor list: build for rcut=6.0 with a 0.5 skin, rebuild every 10 steps
    nlist = NeighborList(rcut=6.0, skin=0.5, box=box, nstlist=10)
    step = {"i": 0}

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
        nlist.maybe_update(x, step["i"])
        step["i"] += 1
        return water_forces(
            x,
            types,
            molecules,
            params,
            box=box,
            rcut_lj=6.0,
            rcut_coulomb=None,
            pairs=nlist.pairs,
        )

    dt = 5e-4
    sim = MDSim(x0, v0, m, force_fn, dt)

    E0 = None
    for t in range(3000):
        stats = sim.step()
        sim.x = wrap_positions(sim.x, box)
        if E0 is None:
            E0 = stats["E"]
        if t % 100 == 0:
            print(
                f"step {t:5d}  K={stats['K']:.8f}  U={stats['U']:.8f}  E={stats['E']:.8f}"
            )

    print(f"Relative energy drift = {abs(stats['E'] - E0)/abs(E0):.2e}")


if __name__ == "__main__":
    main()
