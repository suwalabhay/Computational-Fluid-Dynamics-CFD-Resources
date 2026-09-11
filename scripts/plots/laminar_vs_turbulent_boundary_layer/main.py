"""Compare normalised laminar and turbulent boundary-layer velocity profiles.

The laminar profile is the quadratic (Pohlhausen-type) approximation
u/U = 2 eta - eta^2 and the turbulent profile is the empirical 1/7 power law
u/U = eta^(1/7), both plotted against eta = y / delta.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_POINTS = 200


def laminar_profile(eta):
    """Quadratic laminar approximation u/U_inf = 2 eta - eta^2."""
    return 2 * eta - eta**2


def turbulent_profile(eta):
    """One-seventh power law u/U_inf = eta^(1/7)."""
    return eta ** (1 / 7)


def make_figure(n_points=N_POINTS):
    eta = np.linspace(0, 1, n_points)  # y / delta
    u_laminar = laminar_profile(eta)
    u_turbulent = turbulent_profile(eta)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u_laminar, eta, label="Laminar BL (quadratic)", linewidth=2)
    ax.plot(u_turbulent, eta, label="Turbulent BL (1/7 power law, fuller)", linewidth=2)

    ax.set_title("Laminar vs. Turbulent Boundary Layer Velocity Profiles", fontsize=14)
    ax.set_xlabel(r"Velocity ($u / U_\infty$)", fontsize=12)
    ax.set_ylabel(r"Normalised wall distance ($y / \delta$)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True)

    # Annotations anchored on the curves
    eta_lam, eta_turb = 0.4, 0.02
    ax.annotate(
        "Laminar: gradual rise,\nlower near-wall momentum",
        xy=(laminar_profile(eta_lam), eta_lam),
        xytext=(0.08, 0.6),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )
    ax.annotate(
        "Turbulent: fuller,\nsteep at the wall",
        xy=(turbulent_profile(eta_turb), eta_turb),
        xytext=(0.8, 0.04),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="do not open the plot window"
    )
    parser.add_argument(
        "--output", metavar="DIR", help="save the figure as a PNG file in DIR"
    )
    args = parser.parse_args(argv)

    fig = make_figure()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / "laminar_vs_turbulent_boundary_layer.png",
            dpi=100,
            bbox_inches="tight",
        )
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
