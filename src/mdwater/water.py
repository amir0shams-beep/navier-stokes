import numpy as np


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


# ---------- Intramolecular forces: bonds + angle ----------
def bond_angle_forces(
    x, molecules, k_bond=300.0, r_eq=0.9572, k_theta=50.0, theta_eq_deg=104.52
):
    """
    Harmonic bonds (H-O) and harmonic angle (H-O-H).
    Returns (forces, potential).
    """
    N = x.shape[0]
    f = np.zeros_like(x)
    U = 0.0
    theta_eq = np.deg2rad(theta_eq_deg)
    eps = 1e-12

    for h1, o, h2 in molecules:
        # --- bonds: (1/2) k (|r| - r_eq)^2
        for h in (h1, h2):
            rij = x[h] - x[o]
            r = np.linalg.norm(rij) + eps
            dr = r - r_eq
            U += 0.5 * k_bond * dr * dr
            # Force on h: -k * (r - r_eq) * r_hat
            F = -k_bond * dr * (rij / r)
            f[h] += F
            f[o] -= F

        # --- angle: (1/2) k_theta (theta - theta_eq)^2
        a = x[h1] - x[o]
        b = x[h2] - x[o]
        ra = np.linalg.norm(a) + eps
        rb = np.linalg.norm(b) + eps
        cos_th = np.clip(np.dot(a, b) / (ra * rb), -1.0, 1.0)
        th = np.arccos(cos_th)
        dth = th - theta_eq
        U += 0.5 * k_theta * dth * dth

        sin_th = np.sqrt(max(1.0 - cos_th * cos_th, 1e-14))
        # dtheta/da and dtheta/db
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


# ---------- Intermolecular forces: Coulomb + O-O LJ ----------
def coulomb_and_OO_lj(
    x,
    types,
    molecules,
    q_H=+0.4238,
    q_O=-0.8476,
    kC=1.0,
    epsilon_OO=0.2,
    sigma_OO=3.166,
    rcut=None,
    box=None,
):
    """
    Interactions between atoms of DIFFERENT molecules only:
      - Coulomb for all pairs (i,j) with mol(i) != mol(j)
      - Lennard-Jones only for O-O pairs
    Units arbitrary. kC, epsilon, sigma set scales.
    """
    N = x.shape[0]
    f = np.zeros_like(x)
    U = 0.0

    # map atom -> mol id
    atom2mol = {}
    for mid, (h1, o, h2) in enumerate(molecules):
        atom2mol[h1] = mid
        atom2mol[o] = mid
        atom2mol[h2] = mid

    # charges by type
    q = np.where(types == "O", q_O, q_H)

    if box is not None:
        boxv = np.array([box] * 3) if np.isscalar(box) else np.asarray(box)

    for i in range(N - 1):
        for j in range(i + 1, N):
            if atom2mol[i] == atom2mol[j]:
                continue  # skip intra-molecular here (handled by bonds/angle)

            rij = x[j] - x[i]
            if box is not None:
                rij -= boxv * np.round(rij / boxv)  # minimum image
            r2 = np.dot(rij, rij)
            if r2 == 0.0:
                continue
            if rcut is not None and r2 > rcut * rcut:
                continue

            r = np.sqrt(r2)
            inv_r3 = 1.0 / (r2 * r)

            # Coulomb
            Uc = kC * q[i] * q[j] / r
            Fc = kC * q[i] * q[j] * inv_r3 * rij

            # O-O Lennard-Jones
            Ulj = 0.0
            Flj = 0.0
            if types[i] == "O" and types[j] == "O":
                inv_r2 = 1.0 / r2
                sr2 = (sigma_OO * sigma_OO) * inv_r2
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6
                Ulj = 4.0 * epsilon_OO * (sr12 - sr6)
                coef = 24.0 * epsilon_OO * (2.0 * sr12 - sr6) * inv_r2
                Flj = coef * rij

            f[i] -= Fc + Flj
            f[j] += Fc + Flj
            U += Uc + Ulj

    return f, U


# ---------- Combined water force ----------
def water_forces(x, types, molecules, params, box=None, rcut=None):
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
        rcut=rcut,
        box=box,
    )
    return f1 + f2, U1 + U2
