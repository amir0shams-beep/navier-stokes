import numpy as np
from mdwater.sim import MDSim
from mdwater.forces import lennard_jones


def init_cubic_lattice(n_side=3, spacing=1.5):
    pts = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                pts.append([i * spacing, j * spacing, k * spacing])
    return np.array(pts, float)


def main():
    x0 = init_cubic_lattice(3, 1.3)[:20]  # ~20 particles
    N = len(x0)
    v0 = 0.1 * (np.random.randn(N, 3) - 0.5)  # small random velocities
    m = np.ones(N)  # unit mass
    dt = 0.002
    sim = MDSim(
        x0, v0, m, lennard_jones, dt, {"epsilon": 1.0, "sigma": 1.0, "rcut": 3.0}
    )
    for t in range(2000):
        stats = sim.step()
        if t % 100 == 0:
            print(
                f"step {t:5d}  K={stats['K']:.4f}  U={stats['U']:.4f}  E={stats['E']:.4f}"
            )


if __name__ == "__main__":
    main()
