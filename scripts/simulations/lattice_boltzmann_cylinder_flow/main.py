"""2D flow past a cylinder with the D2Q9 BGK lattice Boltzmann method.

A uniform inflow (Zou/He velocity inlet) passes a circular obstacle with
bounce-back walls; the domain is periodic in y and has a zero-gradient outlet.
The velocity magnitude is animated with Matplotlib and shows the von Karman
vortex street once shedding sets in.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Simulation parameters (lattice units: dx = dt = 1)
REYNOLDS_NUMBER = 350.0  # Re = U * r / nu, based on the cylinder radius
LATTICE_DIMENSIONS = (1040, 360)  # (nx, ny) lattice nodes
NUM_POPULATIONS = 9  # D2Q9
CYLINDER_COORDS = (LATTICE_DIMENSIONS[0] // 4, LATTICE_DIMENSIONS[1] // 2)
CYLINDER_RADIUS = LATTICE_DIMENSIONS[1] // 12
VELOCITY_LATTICE_UNITS = 0.06  # inflow speed U (Mach number U / c_s = 0.10)
STEPS_PER_FRAME = 10  # LBM time steps between animation frames
N_FRAMES = (
    3000  # default number of frames (30 000 time steps; shedding starts near 20 000)
)

# Relaxation parameter omega = 1 / tau with nu = c_s^2 (tau - 1/2), c_s^2 = 1/3
VISCOSITY_LATTICE_UNITS = VELOCITY_LATTICE_UNITS * CYLINDER_RADIUS / REYNOLDS_NUMBER
RELAXATION_PARAMETER = 1.0 / (3.0 * VISCOSITY_LATTICE_UNITS + 0.5)

# Lattice constants
LATTICE_VELOCITIES = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])
LATTICE_WEIGHTS = np.ones(NUM_POPULATIONS) / 36.0
LATTICE_WEIGHTS[np.linalg.norm(LATTICE_VELOCITIES, axis=1) < 1.1] = 1.0 / 9.0
LATTICE_WEIGHTS[0] = 4.0 / 9.0
NOSLIP = np.array(
    [
        LATTICE_VELOCITIES.tolist().index((-LATTICE_VELOCITIES[i]).tolist())
        for i in range(NUM_POPULATIONS)
    ]
)
INDICES_RIGHT_WALL = np.arange(NUM_POPULATIONS)[LATTICE_VELOCITIES[:, 0] < 0]
INDICES_VERTICAL_MIDDLE = np.arange(NUM_POPULATIONS)[LATTICE_VELOCITIES[:, 0] == 0]
INDICES_LEFT_WALL = np.arange(NUM_POPULATIONS)[LATTICE_VELOCITIES[:, 0] > 0]


def compute_density(fin):
    """Zeroth moment: rho = sum_i f_i."""
    return np.sum(fin, axis=0)


def compute_velocity(fin, rho):
    """First moment: u = sum_i c_i f_i / rho."""
    return np.einsum("qd,qxy->dxy", LATTICE_VELOCITIES, fin) / rho


def equilibrium(rho, u):
    """Second-order D2Q9 equilibrium distribution."""
    cu = 3.0 * np.einsum("qd,dxy->qxy", LATTICE_VELOCITIES, u)
    usqr = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho * LATTICE_WEIGHTS[:, None, None] * (1.0 + cu + 0.5 * cu**2 - usqr)


def make_obstacle():
    """Boolean mask of the lattice nodes inside the cylinder."""
    return np.fromfunction(
        lambda x, y: (
            (x - CYLINDER_COORDS[0]) ** 2 + (y - CYLINDER_COORDS[1]) ** 2
            < CYLINDER_RADIUS**2
        ),
        LATTICE_DIMENSIONS,
    )


def make_inflow_velocity():
    """Uniform inflow with a tiny sinusoidal perturbation that triggers shedding."""
    return np.fromfunction(
        lambda d, x, y: (
            (1 - d)
            * VELOCITY_LATTICE_UNITS
            * (1.0 + 1e-4 * np.sin(y / (LATTICE_DIMENSIONS[1] - 1.0) * 2 * np.pi))
        ),
        (2, *LATTICE_DIMENSIONS),
    )


def lbm_step(fin, obstacle, inflow_velocity):
    """Advance the populations by one time step in place; return the velocity."""
    # Right wall: zero-gradient outflow for the populations entering the domain.
    fin[INDICES_RIGHT_WALL, -1, :] = fin[INDICES_RIGHT_WALL, -2, :]

    rho = compute_density(fin)
    u = compute_velocity(fin, rho)

    # Left wall: prescribed velocity, density from the known populations.
    u[:, 0, :] = inflow_velocity[:, 0, :]
    rho[0, :] = (
        compute_density(fin[INDICES_VERTICAL_MIDDLE, 0, :])
        + 2.0 * compute_density(fin[INDICES_RIGHT_WALL, 0, :])
    ) / (1.0 - u[0, 0, :])

    feq = equilibrium(rho, u)
    # Left wall: Zou/He bounce-back of the non-equilibrium part for unknown populations.
    opposite = NOSLIP[INDICES_LEFT_WALL]
    fin[INDICES_LEFT_WALL, 0, :] = (
        feq[INDICES_LEFT_WALL, 0, :] + fin[opposite, 0, :] - feq[opposite, 0, :]
    )

    # BGK collision.
    fout = fin - RELAXATION_PARAMETER * (fin - feq)

    # Full-way bounce-back inside the obstacle (no-slip wall).
    fout[:, obstacle] = fin[NOSLIP][:, obstacle]

    # Streaming (periodic in y).
    for i in range(NUM_POPULATIONS):
        fin[i] = np.roll(fout[i], shift=tuple(LATTICE_VELOCITIES[i]), axis=(0, 1))
    return u


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument("--output", type=Path, help="directory to save the final frame")
    parser.add_argument(
        "--steps",
        type=int,
        default=N_FRAMES,
        help=f"animation frames, each {STEPS_PER_FRAME} LBM time steps "
        f"(default: {N_FRAMES})",
    )
    args = parser.parse_args(argv)

    print(
        f"Re = {REYNOLDS_NUMBER:g}, nu = {VISCOSITY_LATTICE_UNITS:.5f}, "
        f"omega = {RELAXATION_PARAMETER:.4f}, tau = {1 / RELAXATION_PARAMETER:.4f}"
    )

    obstacle = make_obstacle()
    inflow_velocity = make_inflow_velocity()
    fin = equilibrium(1.0, inflow_velocity)
    state = {"time_step": 0}

    fig, ax = plt.subplots(facecolor="black")
    ax.set_facecolor("black")
    initial_speed = np.sqrt(inflow_velocity[0] ** 2 + inflow_velocity[1] ** 2)
    initial_speed[obstacle] = np.nan  # obstacle drawn in the background colour
    image = ax.imshow(
        initial_speed.T,
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=2.0 * VELOCITY_LATTICE_UNITS,
    )
    colorbar = fig.colorbar(image, label="Velocity Magnitude (lattice units)", ax=ax)
    colorbar.ax.yaxis.set_tick_params(color="white")
    colorbar.ax.yaxis.label.set_color("white")
    plt.setp(plt.getp(colorbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("X (lattice units)", color="white")
    ax.set_ylabel("Y (lattice units)", color="white")
    title = ax.set_title("2D Flow Around a Cylinder", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    def update(frame):
        for _ in range(STEPS_PER_FRAME):
            u = lbm_step(fin, obstacle, inflow_velocity)
        state["time_step"] += STEPS_PER_FRAME
        speed = np.sqrt(u[0] ** 2 + u[1] ** 2)
        speed[obstacle] = np.nan
        image.set_data(speed.T)
        title.set_text(f"2D Flow Around a Cylinder (step {state['time_step']})")
        return (image,)

    if args.no_show:
        for frame in range(args.steps):
            update(frame)
    else:
        animation = FuncAnimation(  # noqa: F841 (keep a reference while showing)
            fig, update, frames=args.steps, init_func=lambda: (), repeat=False
        )
        plt.show()

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.output / "lattice_boltzmann_cylinder_flow.png",
            dpi=100,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
