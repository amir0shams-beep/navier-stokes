import numpy as np


def lennard_jones(x, epsilon=1.0, sigma=1.0, rcut=None, box=None):
    """
    x: (N,3) positions
    box: None or float or array-like length 3 (minimum-image if given)
    returns (forces (N,3), potential_energy)
    """
    N = x.shape[0]
    f = np.zeros_like(x)
    U = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = x[j] - x[i]
            if box is not None:
                boxv = np.array([box] * 3) if np.isscalar(box) else np.asarray(box)
                rij -= boxv * np.round(rij / boxv)  # minimum image
            r2 = np.dot(rij, rij)
            if r2 == 0.0:
                continue
            if rcut is not None and r2 > rcut * rcut:
                continue
            inv_r2 = 1.0 / r2
            sr2 = (sigma * sigma) * inv_r2
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            U_ij = 4.0 * epsilon * (sr12 - sr6)
            U += U_ij
            # F_ij = 24ε * (2 sr12 - sr6) * rij / r^2
            coef = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
            fij = coef * rij
            f[i] -= fij
            f[j] += fij
    return f, U
