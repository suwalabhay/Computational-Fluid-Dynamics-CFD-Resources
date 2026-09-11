"""Free-particle 2D time-dependent Schrödinger equation.

A normalised Gaussian wavepacket is evolved with the Strang split-step
Fourier method (hbar = m = 1) on a periodic grid, and the probability density
|psi|^2 is animated as a 3D surface.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Parameters (dimensionless units with hbar = m = 1)
DOMAIN_LENGTH = 10.0  # side length L of the periodic square domain
N_POINTS = 100  # grid points per side
TIME_STEP = 0.01  # dt
FINAL_TIME = 1.0  # total simulated time for the default run
SPEED_FACTOR = 5  # time steps per animation frame
N_FRAMES = int(round(FINAL_TIME / TIME_STEP)) // SPEED_FACTOR


def make_grid(length=DOMAIN_LENGTH, n=N_POINTS):
    """Return the grid spacing and periodic coordinate arrays X, Y."""
    dx = length / n
    x = -length / 2 + dx * np.arange(n)  # periodic grid: no duplicated end point
    X, Y = np.meshgrid(x, x)
    return dx, X, Y


def potential(X, Y):
    """Potential energy V(x, y); zero everywhere for a free particle."""
    return np.zeros_like(X)


def initial_wavefunction(X, Y, dx):
    """Gaussian wavepacket exp(-(x^2 + y^2)/2), normalised so sum |psi|^2 dA = 1."""
    psi = np.exp(-(X**2 + Y**2) / 2).astype(complex)
    return psi / np.sqrt(probability_norm(psi, dx))


def squared_wavenumbers(n, dx):
    """Return k^2 = kx^2 + ky^2 on the FFT grid."""
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(k, k)
    return KX**2 + KY**2


def evolve(psi, dt, k2, V):
    """Advance psi by one Strang step: half kinetic, full potential, half kinetic."""
    half_kinetic = np.exp(-1j * k2 * dt / 4)  # exp(-i (k^2/2) (dt/2))
    psi = np.fft.ifft2(np.fft.fft2(psi) * half_kinetic)
    psi = psi * np.exp(-1j * V * dt)
    return np.fft.ifft2(np.fft.fft2(psi) * half_kinetic)


def probability_norm(psi, dx):
    """Total probability: sum of |psi|^2 times the cell area."""
    return float(np.sum(np.abs(psi) ** 2) * dx**2)


def style_axes(ax, z_max, time):
    ax.set_facecolor("black")
    ax.set_xlim(-DOMAIN_LENGTH / 2, DOMAIN_LENGTH / 2)
    ax.set_ylim(-DOMAIN_LENGTH / 2, DOMAIN_LENGTH / 2)
    ax.set_zlim(0, z_max)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.set_zlabel(r"$|\psi|^2$", color="white")
    ax.set_title(
        f"Time Evolution of Schrödinger Equation in 2D (t = {time:.2f})", color="white"
    )
    for axis in ("x", "y", "z"):
        ax.tick_params(axis=axis, colors="white")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--output", type=Path, help="directory to save the final frame")
    parser.add_argument(
        "--steps",
        type=int,
        default=N_FRAMES,
        help=f"animation frames, each {SPEED_FACTOR} time steps (default: {N_FRAMES})",
    )
    args = parser.parse_args(argv)

    dx, X, Y = make_grid()
    V = potential(X, Y)
    k2 = squared_wavenumbers(N_POINTS, dx)
    state = {"psi": initial_wavefunction(X, Y, dx), "time": 0.0}
    initial_norm = probability_norm(state["psi"], dx)

    density = np.abs(state["psi"]) ** 2
    z_max = density.max()  # fixed colour and z range: the peak only decreases

    fig = plt.figure(facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, density, cmap="viridis", vmin=0, vmax=z_max)
    style_axes(ax, z_max, state["time"])

    cax = fig.add_axes([0.05, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, z_max))
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")

    def update(frame):
        for _ in range(SPEED_FACTOR):
            state["psi"] = evolve(state["psi"], TIME_STEP, k2, V)
            state["time"] += TIME_STEP
        density = np.abs(state["psi"]) ** 2
        ax.clear()
        ax.plot_surface(X, Y, density, cmap="viridis", vmin=0, vmax=z_max)
        style_axes(ax, z_max, state["time"])
        return ()

    if args.no_show:
        for frame in range(args.steps):
            update(frame)
    else:
        animation = FuncAnimation(  # noqa: F841 (keep a reference while showing)
            fig, update, frames=args.steps, init_func=lambda: (), repeat=False
        )
        plt.show()

    final_norm = probability_norm(state["psi"], dx)
    print(
        f"t = {state['time']:.2f}: total probability {final_norm:.15f} "
        f"(change {final_norm - initial_norm:+.2e})"
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.output / "schroedinger_equation.png",
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
