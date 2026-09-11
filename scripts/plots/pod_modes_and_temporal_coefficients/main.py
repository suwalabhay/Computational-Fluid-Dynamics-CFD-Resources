"""First three POD spatial modes and their temporal coefficients for a synthetic 2-D field.

A space-time field u(x, y, t) on a 50 x 30 x 100 grid is built from three
separable structures with decreasing amplitude plus seeded noise. The field
is reshaped into an (Nx Ny) x Nt snapshot matrix, the temporal mean is
removed, and the SVD gives the spatial modes (contour plots) and temporal
coefficients a_i(t) = sigma_i psi_i(t) (time series).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

N_X, N_Y, N_T = 50, 30, 100  # grid points in x, y and number of snapshots
X_RANGE = (1700.0, 2000.0)  # streamwise extent [mm]
Y_RANGE = (0.0, 100.0)  # wall-normal extent [mm]
T_END = 4.0  # duration of the record [s]
# Separable structures A * sin(p pi xi) sin(q pi eta) * g(t), xi, eta in [0, 1]
STRUCTURES = [
    (1.0, 1, 1, lambda t: np.sin(np.pi * t)),
    (0.6, 2, 1, lambda t: np.sin(2 * np.pi * t)),
    (0.3, 1, 2, lambda t: np.cos(3 * np.pi * t)),
]
NOISE_STD = 0.02  # standard deviation of the random fluctuations [-]
SEED = 42  # random seed
N_MODES = 3  # number of modes to plot


def generate_field(noise_std=NOISE_STD, seed=SEED):
    """Return x, y, t and the field u with shape (N_X, N_Y, N_T)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(*X_RANGE, N_X)
    y = np.linspace(*Y_RANGE, N_Y)
    t = np.linspace(0.0, T_END, N_T)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    xi = (X - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])
    eta = (Y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
    field = np.zeros_like(X)
    for amplitude, p, q, g in STRUCTURES:
        field += amplitude * np.sin(p * np.pi * xi) * np.sin(q * np.pi * eta) * g(T)
    field += noise_std * rng.standard_normal(field.shape)
    return x, y, t, field


def pod(field, n_modes=N_MODES):
    """SVD-based POD of the mean-subtracted (Nx Ny) x Nt snapshot matrix.

    Returns the spatial modes reshaped to (N_X, N_Y, n_modes), the temporal
    coefficients a_i(t) = sigma_i psi_i(t) with shape (n_modes, N_T), and the
    fraction of energy in every mode.
    """
    nx, ny, nt = field.shape
    snapshots = field.reshape(nx * ny, nt)
    snapshots = snapshots - snapshots.mean(axis=1, keepdims=True)
    Phi, S, PsiT = np.linalg.svd(snapshots, full_matrices=False)
    modes = Phi[:, :n_modes].reshape(nx, ny, n_modes)
    time_coeffs = S[:n_modes, None] * PsiT[:n_modes, :]
    energy_fraction = S**2 / np.sum(S**2)
    return modes, time_coeffs, energy_fraction


def plot_modes(x, y, t, modes, time_coeffs, energy_fraction):
    """Contour plots of the modes (left) and their time coefficients (right)."""
    n_modes = modes.shape[2]
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(n_modes, 2, width_ratios=[1, 1])
    for i in range(n_modes):
        title = f"Mode {i + 1} ({100 * energy_fraction[i]:.1f}% TKE)"
        ax = fig.add_subplot(gs[i, 0])
        c = ax.contourf(x, y, modes[:, :, i].T, cmap="jet", levels=50)
        fig.colorbar(c, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, time_coeffs[i, :])
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"$a_{i + 1}(t)$")
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="do not open a plot window"
    )
    parser.add_argument(
        "--output", type=Path, metavar="DIR", help="save the figure as PNG in DIR"
    )
    args = parser.parse_args(argv)

    x, y, t, field = generate_field()
    modes, time_coeffs, energy_fraction = pod(field)
    for i in range(N_MODES + 1):
        print(f"mode {i + 1}: TKE = {100 * energy_fraction[i]:.2f} %")

    fig = plot_modes(x, y, t, modes, time_coeffs, energy_fraction)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.output / "pod_modes_and_temporal_coefficients.png",
            dpi=100,
            bbox_inches="tight",
        )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
