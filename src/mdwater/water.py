import numpy as np

try:
    from .ewald import ewald_coulomb_forces_energy
except Exception:
    ewald_coulomb_forces_energy = None  # fallback if not available


try:
    from .pbc import minimum_image
except Exception:

    def _as_boxv(box):
        return (
            np.array([box] * 3, float) if np.isscalar(box) else np.asarray(box, float)
        )

    def minimum_image(rij, box):
        if box is None:
            return rij
        b = _as_boxv(box)
        return rij - b * np.round(rij / b)


# ---------- Topology helpers ----------
def build_one_water_geometry(bond=0.9572, angle_deg=104.52):
    """
    Returns positions (3,3) for [H1, O, H2] centered roughly at O.
    Units are arbitrary; if you use Angstrom here, keep consistent elsewhere.
    """
    theta = np.deg2rad(angle_deg)
    # Put O at origin, H's in x-y plane symmetrically
    O = np.array([0.0, 0.0, 0.0])
    # place H's so |OH|=bond and angle H-O-H = theta
    x = bond * np.sin(theta / 2.0)
    y = bond * np.cos(theta / 2.0)
    H1 = np.array([+x, +y, 0.0])
    H2 = np.array([-x, +y, 0.0])
    return np.stack([H1, O, H2], axis=0)  # order: [H1, O, H2]


def place_waters_grid(n_side=2, spacing=3.0, bond=0.9572, angle_deg=104.52):
    """
    Build a small grid of waters. Returns:
      x: (N,3) positions
      types: (N,) array with strings "H" or "O"
      molecules: list of tuples (h1_idx, o_idx, h2_idx)
    """
    base = build_one_water_geometry(bond=bond, angle_deg=angle_deg)
    types = []
    coords = []
    molecules = []
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                shift = np.array([i * spacing, j * spacing, k * spacing])
                R = base + shift
                # indices in global list
                h1 = idx + 0
                o = idx + 1
                h2 = idx + 2
                coords.append(R[0])
                types.append("H")
                coords.append(R[1])
                types.append("O")
                coords.append(R[2])
                types.append("H")
                molecules.append((h1, o, h2))
                idx += 3
    return np.array(coords, float), np.array(types, dtype=object), molecules


# ---------- Intramolecular forces: bonds + angle (PBC-aware) ----------
def bond_angle_forces(
    x, molecules, k_bond=300.0, r_eq=0.9572, k_theta=50.0, theta_eq_deg=104.52, box=None
):
    """
    Harmonic bonds (H-O) and harmonic angle (H-O-H), using minimum-image vectors
    when a periodic box is provided.
    Returns (forces, potential).
    """
    N = x.shape[0]
    f = np.zeros_like(x)
    U = 0.0
    theta_eq = np.deg2rad(theta_eq_deg)
    eps = 1e-12

    for h1, o, h2 in molecules:
        # --- bonds: (1/2) k (|r| - r_eq)^2 with minimum image for H-O vectors
        for h in (h1, h2):
            rij = minimum_image(x[h] - x[o], box)
            r = np.linalg.norm(rij) + eps
            dr = r - r_eq
            U += 0.5 * k_bond * dr * dr
            # Force on h: -k * (r - r_eq) * r_hat
            F = -k_bond * dr * (rij / r)
            f[h] += F
            f[o] -= F

        # --- angle: (1/2) k_theta (theta - theta_eq)^2
        # Use minimum-image vectors for H1-O and H2-O to measure the angle
        a = minimum_image(x[h1] - x[o], box)
        b = minimum_image(x[h2] - x[o], box)
        ra = np.linalg.norm(a) + eps
        rb = np.linalg.norm(b) + eps

        cos_th = np.clip(np.dot(a, b) / (ra * rb), -1.0, 1.0)
        th = np.arccos(cos_th)
        dth = th - theta_eq
        U += 0.5 * k_theta * dth * dth

        # Derivatives for angle forces
        sin_th = np.sqrt(max(1.0 - cos_th * cos_th, 1e-14))
        # dcos/da and dcos/db
        dcos_da = (b / (ra * rb)) - (cos_th / (ra * ra)) * a
        dcos_db = (a / (ra * rb)) - (cos_th / (rb * rb)) * b
        dth_da = -(1.0 / sin_th) * dcos_da
        dth_db = -(1.0 / sin_th) * dcos_db
        # Forces: -dV/dx = -k_theta * dth * dtheta/dx
        Fa = -k_theta * dth * dth_da
        Fb = -k_theta * dth * dth_db
        Fo = -(Fa + Fb)

        f[h1] += Fa
        f[h2] += Fb
        f[o] += Fo

    return f, U


# ---------- Intermolecular forces: Coulomb + O-O LJ (PBC-aware) ----------
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


# ---------- Combined water force ----------
def water_forces(
    x, types, molecules, params, box=None, rcut_lj=None, rcut_coulomb=None
):
    """
    params: dict with keys:
      k_bond, r_eq, k_theta, theta_eq_deg,
      q_H, q_O, kC,
      epsilon_OO, sigma_OO,
      coulomb_method ('direct'|'ewald'),
      ewald_alpha, ewald_r_real, ewald_kmax
    Returns (forces, potential)
    """
    # 1) Intramolecular bonds + angle (PBC-aware)
    f_ba, U_ba = bond_angle_forces(
        x,
        molecules,
        k_bond=params.get("k_bond", 300.0),
        r_eq=params.get("r_eq", 0.9572),
        k_theta=params.get("k_theta", 50.0),
        theta_eq_deg=params.get("theta_eq_deg", 104.52),
        box=box,
    )

    # 2) Intermolecular O–O Lennard-Jones (shifted at rcut_lj)
    #    (We reuse coulomb_and_OO_lj with charges zeroed to get only LJ.)
    epsilon_OO = params.get("epsilon_OO", 0.2)
    sigma_OO = params.get("sigma_OO", 3.166)

    f_lj, U_lj = coulomb_and_OO_lj(
        x,
        types,
        molecules,
        q_H=0.0,
        q_O=0.0,  # zero-out charges => LJ-only
        kC=0.0,
        epsilon_OO=epsilon_OO,
        sigma_OO=sigma_OO,
        rcut_lj=rcut_lj,
        rcut_coulomb=None,
        box=box,
    )

    # 3) Coulomb: either 'direct' (previous method) or 'ewald' under PBC
    coulomb_method = params.get("coulomb_method", "direct").lower()
    q_H = params.get("q_H", +0.4238)
    q_O = params.get("q_O", -0.8476)

    if coulomb_method == "ewald":
        if ewald_coulomb_forces_energy is None:
            raise RuntimeError("Ewald module not available; did you create ewald.py?")
        if box is None:
            raise ValueError("Ewald requires a periodic box (box != None).")

        # build charges (include intra + inter; water is neutral overall)
        q = np.where(types == "O", q_O, q_H)

        alpha = params.get("ewald_alpha", 0.25)
        r_real = params.get("ewald_r_real", 6.0)
        kmax = params.get("ewald_kmax", 6)

        f_coul, U_coul = ewald_coulomb_forces_energy(
            x, q, box=box, alpha=alpha, rcut_real=r_real, kmax=kmax
        )

    else:
        # fallback: direct-space Coulomb only between different molecules (as before)
        f_coul, U_coul = coulomb_and_OO_lj(
            x,
            types,
            molecules,
            q_H=q_H,
            q_O=q_O,
            kC=params.get("kC", 1.0),
            epsilon_OO=0.0,
            sigma_OO=1.0,  # disable LJ in this call
            rcut_lj=None,
            rcut_coulomb=rcut_coulomb,
            box=box,
        )

    # total
    return f_ba + f_lj + f_coul, U_ba + U_lj + U_coul
