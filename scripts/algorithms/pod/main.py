"""Proper Orthogonal Decomposition (POD) of synthetic flow data via the SVD.

Builds a synthetic spatio-temporal field made of three separable structures,
subtracts the temporal mean, computes the thin SVD of the snapshot matrix and
plots the three leading spatial modes next to their temporal coefficients.
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

N_SAMPLES = 100  # number of time snapshots
N_X = 50  # number of spatial points in x
N_Y = 30  # number of spatial points in y
NUM_MODES = 3  # modes to plot


def generate_synthetic_data(
    n_samples: int, n_x: int, n_y: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a field u(x, y, t) of shape (n_x, n_y, n_samples) and its axes.

    The field is the sum of three separable structures with different spatial
    wavenumbers, temporal frequencies and amplitudes, so it has three
    distinct POD modes.
    """
    x = np.linspace(1700, 2000, n_x)  # mm
    y = np.linspace(0, 100, n_y)  # mm
    t = np.linspace(0, 4, n_samples)  # s

    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    data = (
        np.sin(0.02 * X) * np.cos(0.05 * Y) * np.sin(0.5 * T)
        + 0.5 * np.cos(0.04 * X) * np.sin(0.1 * Y) * np.cos(2.0 * T)
        + 0.25 * np.sin(0.06 * X) * np.cos(0.15 * Y) * np.sin(4.0 * T)
    )
    return data, x, y, t


def preprocess_data(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Subtract the temporal mean of every spatial point (row)."""
    return snapshot_matrix - np.mean(snapshot_matrix, axis=1, keepdims=True)


def create_snapshot_matrix(data: np.ndarray) -> np.ndarray:
    """Reshape (n_x, n_y, n_t) data into an (n_x * n_y) x n_t snapshot matrix."""
    return data.reshape(data.shape[0] * data.shape[1], data.shape[2])


class POD:
    def __init__(self, snapshot_matrix: np.ndarray):
        self.snapshot_matrix = snapshot_matrix
        self.modes: np.ndarray = None
        self.singular_values: np.ndarray = None
        self.time_coeffs: np.ndarray = None

    def run(self) -> None:
        """Perform POD with the thin SVD U~ = Phi Sigma Psi^T."""
        centered = preprocess_data(self.snapshot_matrix)
        phi, sigma, psi_t = np.linalg.svd(centered, full_matrices=False)
        self.modes = phi
        self.singular_values = sigma
        self.time_coeffs = psi_t

    def energy_fractions(self) -> np.ndarray:
        """Fraction of the fluctuation energy sigma_i^2 / sum sigma_j^2 per mode."""
        energy = self.singular_values**2
        return energy / energy.sum()


def plot_modes_and_time_coeffs(
    pod: POD, x: np.ndarray, y: np.ndarray, t: np.ndarray, num_modes: int = 3
):
    """Plot the spatial modes (left) and their temporal coefficients (right)."""
    modes_reshaped = pod.modes[:, :num_modes].reshape(x.shape[0], y.shape[0], -1)

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(num_modes, 2, figure=fig)

    for i in range(num_modes):
        ax = fig.add_subplot(gs[i, 0])
        c = ax.contourf(x, y, modes_reshaped[:, :, i].T, cmap="jet", levels=50)
        fig.colorbar(c, ax=ax)
        ax.set_title(f"Mode {i + 1}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    for i in range(num_modes):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, pod.time_coeffs[i, :])
        ax.set_title(f"Mode {i + 1}")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"$a_{i + 1}$")

    fig.tight_layout()
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="do not open the plot window"
    )
    parser.add_argument(
        "--output", metavar="DIR", help="save the figure as a PNG in DIR"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    data, x, y, t = generate_synthetic_data(N_SAMPLES, N_X, N_Y)
    pod = POD(create_snapshot_matrix(data))
    pod.run()

    for i, fraction in enumerate(pod.energy_fractions()[:NUM_MODES], start=1):
        print(f"Mode {i}: {fraction:.2%} of the fluctuation energy")

    fig = plot_modes_and_time_coeffs(pod, x, y, t, num_modes=NUM_MODES)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "pod_modes.png", dpi=100, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
