import numpy as np
import math
from .pbc import minimum_image


def _as_boxv(box):
    return np.array([box] * 3, float) if np.isscalar(box) else np.asarray(box, float)


def ewald_coulomb_forces_energy(
    x,
    q,
    box,
    alpha=0.25,
    rcut_real=6.0,
    kmax=6,
):
    """
    3D Ewald summation (tin-foil boundary) for Coulomb interactions under PBC.
    Returns (forces (N,3), energy float).
    - x: (N,3) positions
    - q: (N,) charges
    - box: float or (3,) box lengths
    - alpha: splitting parameter
    - rcut_real: real-space cutoff (<= L/2 recommended)
    - kmax: max reciprocal index (sum over -kmax..kmax, excluding k=0)
    """
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    N = x.shape[0]
    assert q.shape == (N,)

    L = _as_boxv(box)
    V = float(np.prod(L))

    # sanity: neutral system
    qsum = float(np.sum(q))
    if abs(qsum) > 1e-10:
        raise ValueError(f"Ewald requires overall charge neutrality; sum(q)={qsum}")

    f = np.zeros_like(x)
    U_real = 0.0

    # ---- Real-space sum (short range)
    rcut2 = rcut_real * rcut_real
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = minimum_image(x[j] - x[i], L)
            r2 = float(np.dot(rij, rij))
            if r2 == 0.0 or r2 > rcut2:
                continue
            r = np.sqrt(r2)
            qr = q[i] * q[j]
            erfc_term = math.erfc(alpha * r) / r
            U_real += qr * erfc_term
            # F_ij = q_i q_j * [ erfc(a r)/r^3 + 2a/√π * exp(-a^2 r^2)/r^2 ] * rij
            c = (
                erfc_term / r2
                + (2.0 * alpha / np.sqrt(np.pi)) * np.exp(-((alpha * r) ** 2)) / r
            )
            fij = qr * c * rij
            f[i] -= fij
            f[j] += fij

    # ---- Reciprocal-space sum (long range)
    U_rec = 0.0
    # k-vectors: 2π * (nx/Lx, ny/Ly, nz/Lz), excluding (0,0,0)
    # Precompute k·r for each k on the fly to avoid huge memory.
    # We accumulate S_cos = Σ q cos(k·r), S_sin = Σ q sin(k·r)
    two_pi = 2.0 * np.pi
    for nx in range(-kmax, kmax + 1):
        for ny in range(-kmax, kmax + 1):
            for nz in range(-kmax, kmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                k = np.array([nx / L[0], ny / L[1], nz / L[2]]) * two_pi
                k2 = float(np.dot(k, k))
                if k2 == 0.0:
                    continue
                damp = np.exp(-k2 / (4.0 * alpha * alpha))
                pref = (
                    (4.0 * np.pi / V) * damp / k2
                )  # factor for forces; 0.5*pref for energy

                # Structure factors
                kr = x @ k  # (N,)
                cos_kr = np.cos(kr)
                sin_kr = np.sin(kr)
                S_cos = float(np.sum(q * cos_kr))
                S_sin = float(np.sum(q * sin_kr))

                # Energy
                U_rec += 0.5 * pref * (S_cos * S_cos + S_sin * S_sin)

                # Forces: F_i += pref * q_i [ S_sin*cos(kr_i) - S_cos*sin(kr_i) ] * k
                factor_i = pref * q * (S_sin * cos_kr - S_cos * sin_kr)  # (N,)
                f += factor_i[:, None] * k[None, :]

    # ---- Self-energy correction (tin-foil)
    U_self = -(alpha / np.sqrt(np.pi)) * float(np.sum(q * q))

    U = U_real + U_rec + U_self
    return f, U
