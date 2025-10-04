import numpy as np
from mdwater.sim import MDSim
from mdwater.water import place_waters_grid, water_forces


def main():
    # Build a tiny 2x2x1 grid => 8 waters => 24 atoms
    x0, types, molecules = place_waters_grid(n_side=2, spacing=3.5)

    N = len(x0)
    m = np.where(types == "O", 16.0, 1.0)  # simple masses
    v0 = 0.02 * (np.random.randn(N, 3) - 0.5)

    params = dict(
        k_bond=300.0,
        r_eq=0.9572,
        k_theta=50.0,
        theta_eq_deg=104.52,
        q_H=+0.4238,
        q_O=-0.8476,  # SPC/E-like charges (unitless here)
        kC=1.0,  # Coulomb constant scale (unitless)
        epsilon_OO=0.2,
        sigma_OO=3.166,  # weak O-O LJ in these units
    )

    def force_fn(x, **kw):
        return water_forces(x, types, molecules, params, rcut=6.0, box=None)

    dt = 0.0005  # small because bonds/angles are stiff springs
    sim = MDSim(x0, v0, m, force_fn, dt)

    for t in range(3000):
        stats = sim.step()
        if t % 100 == 0:
            print(
                f"step {t:5d}  K={stats['K']:.8f}  U={stats['U']:.8f}  E={stats['E']:.8f}"
            )


if __name__ == "__main__":
    main()
