"""Animate the 2D wave equation on a square with an explicit leapfrog scheme.

A Gaussian pulse at rest in the centre of [-L, L]^2 spreads outward, reflects
from the fixed (u = 0) edges and interferes with itself. The field is shown as a
3D surface on a dark background.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

L = 5.0  # half-width of the square domain
NX = 100  # grid points in x
NY = 100  # grid points in y
C = 1.0  # wave speed
CFL_SAFETY = 0.5  # dt = CFL_SAFETY * min(dx, dy) / (c * sqrt(2))
T_END = 40.0  # simulated time of the default animation
SCALE_FACTOR = 5.0  # amplitude of the initial Gaussian pulse


def make_grid():
    """Return x, y, X, Y (X, Y from meshgrid, so axis 0 is y), dx, dy, dt."""
    x = np.linspace(-L, L, NX)
    y = np.linspace(-L, L, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = CFL_SAFETY * min(dx, dy) / (C * np.sqrt(2))
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy, dt


def initial_condition(X, Y):
    """Return (u_prev, u): Gaussian pulse with zero initial velocity."""
    u0 = SCALE_FACTOR * np.exp(-0.5 * (X**2 + Y**2))
    return u0.copy(), u0.copy()


def leapfrog_step(u_prev, u, dx, dy, dt):
    """Return (u, u_new) after one leapfrog step with u = 0 on the boundary."""
    u_new = np.zeros_like(u)
    u_new[1:-1, 1:-1] = (
        2 * u[1:-1, 1:-1]
        - u_prev[1:-1, 1:-1]
        + (C * dt) ** 2
        * (
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2  # axis 1 = x
            + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2  # axis 0 = y
        )
    )
    return u, u_new


def style_axes(ax):
    ax.set_zlim(-SCALE_FACTOR, SCALE_FACTOR)
    ax.set_title("2D Wave Equation Simulation Using Finite Differences", color="white")
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("U", color="white")
    for axis in ("x", "y", "z"):
        ax.tick_params(axis=axis, colors="white")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--output", metavar="DIR", help="save the final frame as PNG")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of time steps (frames); default reaches t = 40",
    )
    args = parser.parse_args(argv)

    X, Y, dx, dy, dt = make_grid()
    n_steps = int(T_END / dt) if args.steps is None else args.steps
    state = {"u_prev": None, "u": None, "step": 0}
    state["u_prev"], state["u"] = initial_condition(X, Y)

    # Fixed colour scale so the colour bar stays valid for every frame
    norm = Normalize(vmin=-SCALE_FACTOR, vmax=SCALE_FACTOR)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    surf = ax.plot_surface(X, Y, state["u"], cmap="viridis", norm=norm)
    style_axes(ax)
    color_bar = fig.colorbar(
        surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, location="left"
    )
    color_bar.set_label("Wave Amplitude", color="white")
    color_bar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(color_bar.ax.axes, "yticklabels"), color="white")
    artists = {"surf": surf}

    def advance():
        state["u_prev"], state["u"] = leapfrog_step(
            state["u_prev"], state["u"], dx, dy, dt
        )
        state["step"] += 1

    def redraw():
        artists["surf"].remove()
        artists["surf"] = ax.plot_surface(X, Y, state["u"], cmap="viridis", norm=norm)
        style_axes(ax)
        return (artists["surf"],)

    def update(frame):
        advance()
        return redraw()

    if args.no_show:
        for _ in range(n_steps):
            advance()
        redraw()
    else:
        anim = FuncAnimation(fig, update, frames=n_steps, blit=False, repeat=False)
        plt.show()
        del anim

    print(f"dt = {dt:.4f}, steps = {state['step']}, t = {state['step'] * dt:.2f}")
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / "wave_2d.png",
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
