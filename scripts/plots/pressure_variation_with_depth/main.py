"""Plot the linear increase of hydrostatic pressure with depth, P = P0 + rho g h."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATMOSPHERIC_PRESSURE = 101325.0  # Pa


def hydrostatic_pressure(
    depth, fluid_density, g, surface_pressure=ATMOSPHERIC_PRESSURE
):
    """Absolute pressure (Pa) at ``depth`` (m) below a free surface."""
    return surface_pressure + fluid_density * g * depth


def plot_pressure_variation_with_depth(fluid_density=1000, g=9.81, max_depth=20):
    """Plot pressure against depth.

    fluid_density in kg/m^3, g in m/s^2, max_depth in m. Returns the figure.
    """
    depth = np.linspace(0, max_depth, 100)
    pressure = hydrostatic_pressure(depth, fluid_density, g)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(pressure, depth, label="Pressure vs Depth", color="blue")

    # Invert the y-axis so that depth increases downwards
    plt.gca().invert_yaxis()

    plt.title("Pressure Variation with Depth")
    plt.xlabel("Absolute pressure (Pa)")
    plt.ylabel("Depth (m)")
    plt.grid(True)
    plt.legend()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-show", action="store_true", help="do not open a plot window"
    )
    parser.add_argument(
        "--output", metavar="DIR", help="save the figure as a PNG in DIR"
    )
    args = parser.parse_args(argv)

    fig = plot_pressure_variation_with_depth()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / "pressure_variation_with_depth.png", dpi=100, bbox_inches="tight"
        )
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
