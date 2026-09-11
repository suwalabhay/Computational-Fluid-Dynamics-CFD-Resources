"""Animate the 1D heat equation (Crank-Nicolson) and 1D wave equation (leapfrog).

Both equations start from the same Gaussian pulse on 0 <= x <= L with
homogeneous Dirichlet boundaries. The heat equation is advanced with the
implicit Crank-Nicolson scheme (unconditionally stable), the wave equation with
the explicit second-order leapfrog scheme, whose Courant number c*dt/dx must
not exceed 1. Both share one time step, set by the wave CFL limit.
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

L = 10.0  # domain length (nondimensional)
T = 500.0  # total simulated time
NX = 500  # number of grid points
C = 1.0  # wave speed
D = 1.0  # diffusivity of the heat equation
COURANT = 0.9  # c*dt/dx for the leapfrog scheme (must be <= 1)
PULSE_WIDTH = 5.0  # initial condition exp(-PULSE_WIDTH * (x - L/2)^2)


def make_grid(length=L, nx=NX, c=C, courant=COURANT, total_time=T):
    """Return x, dx, dt and the number of steps needed to reach total_time."""
    x = np.linspace(0.0, length, nx)
    dx = x[1] - x[0]
    nt = int(np.ceil(total_time / (courant * dx / c)))
    dt = total_time / nt  # fits exactly into total_time, Courant <= courant
    return x, dx, dt, nt


def crank_nicolson_operators(nx, r):
    """Return (LU of A, B) for (I - r/2 L) u^{n+1} = (I + r/2 L) u^n.

    L is the second-difference matrix; the first and last rows are identity
    rows so that u = 0 is kept on both boundaries. r = D*dt/dx^2.
    """
    lap = diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)).tolil()
    lap[0, :] = 0.0
    lap[-1, :] = 0.0
    lap = lap.tocsc()
    eye = identity(nx, format="csc")
    a = (eye - 0.5 * r * lap).tocsc()
    b = (eye + 0.5 * r * lap).tocsc()
    return splu(a), b


def heat_step(u, lu, b):
    """Advance the heat solution one Crank-Nicolson step (in place)."""
    u[:] = lu.solve(b @ u)
    u[0] = u[-1] = 0.0


def wave_step(u_prev, u, courant2):
    """Advance the wave solution one leapfrog step (in place)."""
    u_new = np.empty_like(u)
    u_new[1:-1] = (
        2.0 * u[1:-1] - u_prev[1:-1] + courant2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    )
    u_new[0] = u_new[-1] = 0.0
    u_prev[:] = u
    u[:] = u_new


def initial_state(x, dx, dt):
    """Return (heat, wave_prev, wave) for a Gaussian pulse at rest."""
    u0 = np.exp(-PULSE_WIDTH * (x - L / 2) ** 2)
    u0[0] = u0[-1] = 0.0
    courant2 = (C * dt / dx) ** 2
    # Zero initial velocity: u^{-1} = u^0 + (C^2/2) * delta^2 u^0 (second order)
    u_prev = u0.copy()
    u_prev[1:-1] += 0.5 * courant2 * (u0[2:] - 2.0 * u0[1:-1] + u0[:-2])
    return u0.copy(), u_prev, u0.copy()


def setup_figure(x):
    """Create the two-panel dark figure; return (fig, artists)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="black")
    for ax in (ax1, ax2):
        ax.set_facecolor("black")
        ax.set_xlim(0, L)
        ax.grid(False)
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.set_ylabel("Amplitude", fontsize=14, color="white")
    (line_heat,) = ax1.plot(x, np.zeros_like(x), color="cyan", lw=2)
    (line_wave,) = ax2.plot(x, np.zeros_like(x), color="magenta", lw=2)
    ax1.set_title("Heat Equation (Crank-Nicolson)", fontsize=16, color="white")
    ax2.set_title("Wave Equation (Leapfrog)", fontsize=16, color="white")
    ax2.set_xlabel("Spatial Coordinate $x$", fontsize=14, color="white")
    ax1.set_ylim(-0.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    text_kw = dict(fontsize=14, color="white", bbox=dict(facecolor="black", alpha=0.5))
    time_text1 = ax1.text(0.75, 0.85, "", transform=ax1.transAxes, **text_kw)
    time_text2 = ax2.text(0.75, 0.85, "", transform=ax2.transAxes, **text_kw)
    fig.tight_layout()
    return fig, (line_heat, line_wave, time_text1, time_text2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--output", metavar="DIR", help="save the final frame as PNG")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of time steps (animation frames); default reaches t = T",
    )
    args = parser.parse_args(argv)

    x, dx, dt, nt = make_grid()
    n_steps = nt if args.steps is None else args.steps
    r = D * dt / dx**2
    courant2 = (C * dt / dx) ** 2
    print(
        f"dx = {dx:.4f}, dt = {dt:.4f}, Courant = {np.sqrt(courant2):.3f}, "
        f"r = D dt/dx^2 = {r:.2f} (Crank-Nicolson: stable for any r)"
    )
    if courant2 > 1.0:
        print("Warning: Courant number > 1, the leapfrog wave solver is unstable.")

    lu, b = crank_nicolson_operators(len(x), r)
    heat, wave_prev, wave = initial_state(x, dx, dt)

    fig, (line_heat, line_wave, text1, text2) = setup_figure(x)

    def draw(step):
        line_heat.set_ydata(heat)
        line_wave.set_ydata(wave)
        label = f"Time = {step * dt:.4f} s"
        text1.set_text(label)
        text2.set_text(label)
        return [line_heat, line_wave, text1, text2]

    def animate(i):
        heat_step(heat, lu, b)
        wave_step(wave_prev, wave, courant2)
        return draw(i + 1)

    draw(0)
    if args.no_show:
        for _ in range(n_steps):
            heat_step(heat, lu, b)
            wave_step(wave_prev, wave, courant2)
        draw(n_steps)
    else:
        ani = animation.FuncAnimation(
            fig, animate, frames=n_steps, interval=20, blit=True, repeat=False
        )
        plt.show()  # after the window closes, the figure holds the last frame
        del ani

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / "heat_and_wave_1d.png",
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
