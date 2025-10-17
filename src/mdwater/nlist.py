import numpy as np
from .pbc import minimum_image


def _as_boxv(box):
    return np.array([box] * 3, float) if np.isscalar(box) else np.asarray(box, float)


class NeighborList:
    """
    PBC cell-list + Verlet skin. Build pairs within rlist = rcut + skin.
    Rebuild every nstlist steps.
    """

    def __init__(self, rcut, skin=0.5, box=None, nstlist=10):
        self.rcut = float(rcut)
        self.skin = float(skin)
        self.rlist = self.rcut + self.skin
        self.box = _as_boxv(box) if box is not None else None
        self.nstlist = int(nstlist)
        self.step_built = -1
        self.pairs = None

    def maybe_update(self, x, step):
        if self.pairs is None or (step - self.step_built) >= self.nstlist:
            self.update(x)
            self.step_built = step
            return True
        return False

    def update(self, x):
        x = np.asarray(x, float)
        N = x.shape[0]
        if self.box is None:
            raise ValueError("NeighborList currently requires PBC box.")
        L = self.box
        # choose cells so cell_size <= rlist
        ncell = np.maximum(1, np.floor(L / self.rlist).astype(int))
        cell_size = L / ncell
        # assign atoms to cells
        idx = np.floor(x / cell_size).astype(int) % ncell
        # map cell -> list of atom indices
        cells = {}
        for i in range(N):
            key = (idx[i, 0], idx[i, 1], idx[i, 2])
            cells.setdefault(key, []).append(i)
        pairs = []
        # neighbor cells (27)
        for cx in range(ncell[0]):
            for cy in range(ncell[1]):
                for cz in range(ncell[2]):
                    here = cells.get((cx, cy, cz), [])
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                nx = (cx + dx) % ncell[0]
                                ny = (cy + dy) % ncell[1]
                                nz = (cz + dz) % ncell[2]
                                neigh = cells.get((nx, ny, nz), [])
                                for i in here:
                                    for j in neigh:
                                        if j <= i:  # avoid dup/self
                                            continue
                                        rij = minimum_image(x[j] - x[i], L)
                                        if np.dot(rij, rij) <= self.rlist * self.rlist:
                                            pairs.append((i, j))
        self.pairs = np.asarray(pairs, dtype=np.int32)
