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
    pairs=None,  # <-- NEW
):
    """
    3D Ewald (tin-foil). If `pairs` is provided, the real-space sum iterates
    only over those i,j; reciprocal space is unchanged.
    """
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    N = x.shape[0]
    L = np.array([box] * 3, float) if np.isscalar(box) else np.asarray(box, float)
    V = float(np.prod(L))

    if abs(float(np.sum(q))) > 1e-10:
        raise ValueError("Ewald requires overall charge neutrality.")

    f = np.zeros_like(x)
    U_real = 0.0
    rcut2 = rcut_real * rcut_real

    # ---- Real-space
    if pairs is None:
        pair_iter = ((i, j) for i in range(N - 1) for j in range(i + 1, N))
    else:
        pair_iter = pairs

    for i, j in pair_iter:
        rij = minimum_image(x[j] - x[i], L)
        r2 = float(np.dot(rij, rij))
        if r2 == 0.0 or r2 > rcut2:
            continue
        r = np.sqrt(r2)
        qr = q[i] * q[j]
        erfc_term = math.erfc(alpha * r) / r
        U_real += qr * erfc_term
        c = (
            erfc_term / r2
            + (2.0 * alpha / math.sqrt(math.pi)) * math.exp(-((alpha * r) ** 2)) / r
        )
        fij = qr * c * rij
        f[i] -= fij
        f[j] += fij

    # ---- Reciprocal-space
    U_rec = 0.0
    two_pi = 2.0 * math.pi
    for nx in range(-kmax, kmax + 1):
        for ny in range(-kmax, kmax + 1):
            for nz in range(-kmax, kmax + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                k = np.array([nx / L[0], ny / L[1], nz / L[2]]) * two_pi
                k2 = float(np.dot(k, k))
                if k2 == 0.0:
                    continue
                damp = math.exp(-k2 / (4.0 * alpha * alpha))
                pref = (4.0 * math.pi / V) * damp / k2

                kr = x @ k
                cos_kr = np.cos(kr)
                sin_kr = np.sin(kr)
                S_cos = float(np.sum(q * cos_kr))
                S_sin = float(np.sum(q * sin_kr))

                U_rec += 0.5 * pref * (S_cos * S_cos + S_sin * S_sin)

                factor_i = pref * q * (S_sin * cos_kr - S_cos * sin_kr)
                f += factor_i[:, None] * k[None, :]

    # ---- Self term
    U_self = -(alpha / math.sqrt(math.pi)) * float(np.sum(q * q))

    return f, (U_real + U_rec + U_self)
