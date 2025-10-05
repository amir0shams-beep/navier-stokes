import numpy as np
from mdwater.sim import MDSim
from mdwater.water import place_waters_grid, water_forces
from mdwater.pbc import wrap_positions


def main():
    # small system
    x0, types, molecules = place_waters_grid(n_side=2, spacing=3.5)
    N = len(x0)
    m = np.where(types == "O", 16.0, 1.0)
    rng = np.random.default_rng(0)
    v0 = 0.02 * (rng.standard_normal((N, 3)) - 0.5)

    # cubic box and cutoff (must satisfy rcut <= L/2)
    L = 14.0
    box = L

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
        # Use LJ cutoff; do NOT cut Coulomb (small system). Keep rcut_lj <= L/2.
        return water_forces(
            x, types, molecules, params, box=box, rcut_lj=6.0, rcut_coulomb=None
        )

    dt = 5e-4
    sim = MDSim(x0, v0, m, force_fn, dt)

    E0 = None
    for t in range(3000):
        stats = sim.step()
        # wrap positions into the periodic box after each step
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


def coulomb_and_OO_lj(
    x,
    types,
    molecules,
    q_H=+0.4238,
    q_O=-0.8476,
    kC=1.0,
    epsilon_OO=0.2,
    sigma_OO=3.166,
    rcut_lj=None,  # separate cutoff for LJ
    rcut_coulomb=None,  # separate cutoff for Coulomb (None = no cutoff)
    box=None,
):
    """
    Interactions between atoms of DIFFERENT molecules only:
      - Coulomb for all pairs (i,j) with mol(i) != mol(j)
      - Lennard-Jones only for O-O pairs
    Uses a potential-shift for LJ so U(rcut_lj) = 0.
    """
    N = x.shape[0]
    f = np.zeros_like(x)
    U = 0.0

    # atom -> mol id
    atom2mol = {}
    for mid, (h1, o, h2) in enumerate(molecules):
        atom2mol[h1] = mid
        atom2mol[o] = mid
        atom2mol[h2] = mid

    # charges by type
    q = np.where(types == "O", q_O, q_H)

    # Precompute LJ shift at rcut (potential shift only)
    Ulj_shift = 0.0
    if rcut_lj is not None:
        inv_rc2 = 1.0 / (rcut_lj * rcut_lj)
        sr2c = (sigma_OO * sigma_OO) * inv_rc2
        sr6c = sr2c * sr2c * sr2c
        sr12c = sr6c * sr6c
        Ulj_shift = 4.0 * epsilon_OO * (sr12c - sr6c)

    for i in range(N - 1):
        for j in range(i + 1, N):
            if atom2mol[i] == atom2mol[j]:
                continue  # intra-molecular handled by bonds/angle

            rij = minimum_image(x[j] - x[i], box)
            r2 = float(np.dot(rij, rij))
            if r2 == 0.0:
                continue

            r = np.sqrt(r2)
            inv_r2 = 1.0 / r2
            Uc = 0.0
            Fc = 0.0
            Ulj = 0.0
            Flj = 0.0

            # Coulomb (optional cutoff; recommend None for small systems)
            if (rcut_coulomb is None) or (r2 <= rcut_coulomb * rcut_coulomb):
                Uc = kC * q[i] * q[j] / r
                Fc = kC * q[i] * q[j] * (rij / (r2 * r))

            # O-O Lennard-Jones with potential shift
            if types[i] == "O" and types[j] == "O":
                if (rcut_lj is None) or (r2 <= rcut_lj * rcut_lj):
                    sr2 = (sigma_OO * sigma_OO) * inv_r2
                    sr6 = sr2 * sr2 * sr2
                    sr12 = sr6 * sr6
                    Ulj_base = 4.0 * epsilon_OO * (sr12 - sr6)
                    Ulj = Ulj_base - Ulj_shift  # shift so U(rcut_lj) = 0
                    coef = 24.0 * epsilon_OO * (2.0 * sr12 - sr6) * inv_r2
                    Flj = coef * rij  # standard LJ force (no force-shift)

            f[i] -= Fc + Flj
            f[j] += Fc + Flj
            U += Uc + Ulj

    return f, U


def water_forces(
    x, types, molecules, params, box=None, rcut_lj=None, rcut_coulomb=None
):
    """
    params: dict with keys:
      k_bond, r_eq, k_theta, theta_eq_deg, q_H, q_O, kC, epsilon_OO, sigma_OO
    Returns (forces, potential)
    """
    f1, U1 = bond_angle_forces(
        x,
        molecules,
        k_bond=params.get("k_bond", 300.0),
        r_eq=params.get("r_eq", 0.9572),
        k_theta=params.get("k_theta", 50.0),
        theta_eq_deg=params.get("theta_eq_deg", 104.52),
        box=box,
    )
    f2, U2 = coulomb_and_OO_lj(
        x,
        types,
        molecules,
        q_H=params.get("q_H", +0.4238),
        q_O=params.get("q_O", -0.8476),
        kC=params.get("kC", 1.0),
        epsilon_OO=params.get("epsilon_OO", 0.2),
        sigma_OO=params.get("sigma_OO", 3.166),
        rcut_lj=rcut_lj,
        rcut_coulomb=rcut_coulomb,
        box=box,
    )
    return f1 + f2, U1 + U2
