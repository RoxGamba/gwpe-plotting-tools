"""
Make corner plots of bilby/RIFT posteriors.

This module provides a command-line interface for generating corner plots
from bilby and RIFT posterior sample files.

Example usage
-------------
.. code-block:: bash

    python plot_posterior.py -i /path/to/injrec_gr_ecc_abhf_dc.hdf5 \\
                              /path/to/injrec_gr_ecc_abhf_rg.hdf5 \\
                             -p chirp_mass mass_ratio chi_eff \\
                             -r '[27, 29.5]' '[0.25,0.45]' '[0.25, 0.55]' \\
                             -p eccentricity mean_per_ano \\
                             -r '[0.06, 0.14]' '[0, np.pi]' \\
                             -l DC RG

This will make two corner plots, each comparing the results from the two
hdf5 inputs.
"""

import argparse
import os

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pesummary.core.plots.figure import figure
from pesummary.core.plots.publication import _triangle_axes

from .constants import MDC_TO_BILBY
from . import gwutils
from .posteriors import BilbyPosterior, RIFTPosterior

try:
    import palettable

    PALETTABLE_AVAILABLE = True
except ImportError:
    PALETTABLE_AVAILABLE = False


def read_truth(filename, kind="bilby-hdf5"):
    """
    Read truth dictionary from config for injections.

    Reads injection parameters (truth values) from configuration files
    in various formats including bilby ini/json files and RIFT-style
    text files.

    Parameters
    ----------
    filename : str
        Path to the truth/injection file.
    kind : str, optional
        Type of posterior file. Options are 'bilby-hdf5', 'bilby-json',
        or 'rift'. Default is 'bilby-hdf5'.

    Returns
    -------
    dict
        Dictionary mapping parameter names to their true values.

    Raises
    ------
    ValueError
        If the input file format is not recognized.
    """
    pardic = None
    if "bilby" in kind:
        if "ini" in filename:
            with open(filename, "r") as f:
                for line in f:
                    if "injection-dict" not in line:
                        continue
                    pardic = eval(line[15:])
        elif "json" in filename:
            import json

            with open(filename, "r") as f:
                pardic = json.load(f)
    if kind == "rift" or pardic is None:
        injpars = np.genfromtxt(filename, dtype=float, names=True)
        pardic = {MDC_TO_BILBY[key]: injpars[key] for key in injpars.dtype.names}
        if "eccentricity" not in pardic.keys():
            pardic["eccentricity"] = 0.0
        if "mean_per_ano" not in pardic.keys():
            pardic["mean_per_ano"] = 0.0
    if pardic is None:
        raise ValueError("Unrecognized input kind.")

    fillout_pardic(pardic)
    return pardic


def fillout_pardic(pardic):
    """
    Fill out a parameter dictionary with derived parameters.

    Computes derived parameters from basic ones, assuming that the
    mass ratio, spins, theta_jn, and one of total mass or chirp mass
    are already present.

    Parameters
    ----------
    pardic : dict
        Parameter dictionary to fill out. Modified in place.

    Notes
    -----
    This function modifies the input dictionary in place and adds
    derived parameters such as chi_eff, chi_p, symmetric_mass_ratio,
    total_mass, chirp_mass, and individual masses when possible.
    """
    keylist = pardic.keys()
    if "chi_1" not in keylist and "spin_1z" in keylist:
        pardic["chi_1"] = pardic["spin_1z"]
    if "chi_2" not in keylist and "spin_2z" in keylist:
        pardic["chi_2"] = pardic["spin_2z"]
    # Maybe add conversion from spin components to angles for precession...
    if (
        "chi_eff" not in keylist
        and "chi_1" in keylist
        and "chi_2" in keylist
        and "mass_ratio" in keylist
    ):
        pardic["chi_eff"] = (
            pardic["chi_1"] + pardic["mass_ratio"] * pardic["chi_2"]
        ) / (1.0 + pardic["mass_ratio"])
    if (
        "chi_eff" not in keylist
        and "a_1" in keylist
        and "a_2" in keylist
        and "mass_ratio" in keylist
        and "tilt_1" in keylist
        and "tilt_2" in keylist
    ):
        pardic["chi_eff"] = (
            pardic["a_1"] * np.cos(pardic["tilt_1"])
            + pardic["mass_ratio"] * pardic["a_2"] * np.cos(pardic["tilt_2"])
        ) / (1.0 + pardic["mass_ratio"])
    if "symmetric_mass_ratio" not in keylist and "mass_ratio" in keylist:
        pardic["symmetric_mass_ratio"] = (
            pardic["mass_ratio"] / (1.0 + pardic["mass_ratio"]) ** 2
        )
    if (
        "total_mass" not in keylist
        and "chirp_mass" in keylist
        and "symmetric_mass_ratio" in keylist
    ):
        pardic["total_mass"] = pardic["chirp_mass"] * pardic[
            "symmetric_mass_ratio"
        ] ** (-0.6)
    if (
        "chirp_mass" not in keylist
        and "total_mass" in keylist
        and "symmetric_mass_ratio" in keylist
    ):
        pardic["chirp_mass"] = pardic["total_mass"] * pardic[
            "symmetric_mass_ratio"
        ] ** (0.6)
    if "mass_1" not in keylist and "total_mass" in keylist and "mass_ratio" in keylist:
        pardic["mass_1"] = pardic["total_mass"] / (1.0 + pardic["mass_ratio"])
    if "mass_2" not in keylist and "total_mass" in keylist and "mass_ratio" in keylist:
        pardic["mass_2"] = (
            pardic["mass_ratio"] * pardic["total_mass"] / (1.0 + pardic["mass_ratio"])
        )
    if "cos_theta_jn" not in keylist and "theta_jn" in keylist:
        pardic["cos_theta_jn"] = np.cos(pardic["theta_jn"])
    if "chi_p" not in keylist:
        if "tilt_1" in keylist:
            pardic["chi_p"] = gwutils.compute_chi_prec(
                pardic["mass_1"],
                pardic["mass_2"],
                pardic["a_1"],
                pardic["a_2"],
                pardic["tilt_1"],
                pardic["tilt_2"],
            )
        elif (
            "spin_1x" in keylist
            and "spin_1y" in keylist
            and "spin_2x" in keylist
            and "spin_2y" in keylist
        ):
            pardic["chi_p"] = gwutils.compute_chi_prec_from_xyz(
                pardic["mass_ratio"],
                pardic["spin_1x"],
                pardic["spin_1y"],
                pardic["spin_2x"],
                pardic["spin_2y"],
            )
    for par, other in zip(["alpha", "tau"], ["tau", "alpha"]):
        if f"delta_{par}lm0" in keylist:
            if pardic[f"delta_{par}lm0"] != 0.0 and pardic[f"delta_{other}lm0"] == 0.0:
                pardic[f"delta_{other}lm0"] = -1.0 + 1 / (1 + pardic[f"delta_{par}lm0"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Make corner plots of bilby/RIFT posteriors."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        nargs="*",
        help="Path to hdf5 (bilby)/extrinsic_posterior_samples.dat (rift) file(s).",
    )
    parser.add_argument(
        "-t",
        "--truth",
        type=str,
        default=None,
        nargs="*",
        help=(
            "Truth file(s) for injections; if not provided, "
            "looking for config with similar name to posterior file."
        ),
    )
    parser.add_argument(
        "-l", "--label", type=str, nargs="*", default=None, help="Labels for legends."
    )
    parser.add_argument(
        "-p",
        "--par",
        type=str,
        default=None,
        action="append",
        nargs="*",
        help=(
            "For each corner plot, add an instance specifying what "
            "parameters it should plot. E.g.: plot_posterior.py -i res.hdf5 "
            "-p chirp_mass chi_eff -p eccentricity mean_per_ano"
        ),
    )
    parser.add_argument(
        "-r",
        "--range",
        type=str,
        action="append",
        nargs="*",
        help="Parameter ranges for every par in every corner.",
    )
    parser.add_argument(
        "-k",
        "--ticks",
        type=str,
        action="append",
        nargs="*",
        default=None,
        help="Ticks for every par in every corner.",
    )
    parser.add_argument("-s", "--save", action="store_true", help="Save plot(s)?")
    parser.add_argument(
        "-tr",
        "--triangle",
        action="store_true",
        help="Make triangle plots with pesummary (when possible).",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="Name for figures")
    parser.add_argument("--size", type=float, default=3, help="Size of plot")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for saved plots (default: current directory)",
    )

    args = parser.parse_args()

    Np = len(args.input)

    if args.truth is None:
        args.truth = ["" for _ in range(Np)]

    if PALETTABLE_AVAILABLE:
        cmap = palettable.cartocolors.diverging.Tropic_7.mpl_colormap
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "beyond", ["darkturquoise", "aliceblue", "coral"]
        )
    base_cols = cmap(np.array(range(Np)) / (max(Np - 1, 1)))
    # Default colors for up to 4 plots
    DEFAULT_COLORS = [
        (0.2, 0.6, 0.2, 1.0),  # green
        (0.9, 0.5, 0.6, 1.0),  # pink
        (0.2, 0.6, 0.2, 1.0),  # green
        (0.9, 0.5, 0.6, 1.0),  # pink
    ]
    base_cols = DEFAULT_COLORS[:Np] if Np <= len(DEFAULT_COLORS) else base_cols
    line_cols = []
    truth_cols = []
    for col in base_cols:
        line_cols.append(
            "#" + "".join([hex(int(c * 255))[2:].zfill(2) for c in col[:3]])
        )
        truth_cols.append(
            "#" + "".join([hex(int(0.5 * c * 255))[2:].zfill(2) for c in col[:3]])
        )
    if Np == 1 or True:
        truth_cols = ["k"] * Np

    figs = []

    for ji, (infile, truthfile, color, truthcolor) in enumerate(
        zip(args.input, args.truth, line_cols, truth_cols)
    ):

        if ".hdf5" in infile:
            inkind = "bilby-hdf5"
        elif ".json" in infile:
            inkind = "bilby-json"
        elif ".txt" in infile:
            inkind = "txt"
        else:
            inkind = "rift"

        # Create the appropriate posterior object
        if inkind in ["bilby-hdf5", "bilby-json", "bilby-pkl"]:
            post = BilbyPosterior(infile)
        elif inkind == "rift":
            post = RIFTPosterior(infile)
        else:
            raise ValueError(f"Unknown file kind: {inkind}")

        if not truthfile:
            truthfile = infile.replace(
                infile.split("/")[-1].split("_")[0], "config"
            ).replace(".hdf5", ".ini")
            print(f"Looking for truth file: {truthfile}.")
        if not os.path.exists(truthfile):
            print(f"Couldn't find truth file {truthfile}.")
            truths = None
        else:
            truths = read_truth(truthfile, kind=inkind)

        for jp, parlist in enumerate(args.par):
            if args.range is not None:
                lims = [eval(interval) for interval in args.range[jp]]
            else:
                lims = [
                    [
                        np.min(post.__getattribute__(par)),
                        np.max(post.__getattribute__(par)),
                    ]
                    for par in parlist
                ]
            if truths is not None:
                true_vals = [truths[par] for par in parlist]
            else:
                true_vals = None

            if len(parlist) == 2 and args.triangle:
                if jp >= len(figs):
                    fig, ax1, temp, ax2, ax3 = _triangle_axes(
                        figsize=(args.size * 2, args.size * 2)
                    )
                    temp.remove()
                    ax1.set_yticks([])
                    ax1.grid(visible=False)
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                    ax3.grid(visible=False)
                    figs.append(fig)
                post.make_triangle_plot(
                    parlist,
                    color,
                    args.label[ji] if args.label is not None else None,
                    fig=figs[jp],
                    axes=figs[jp].get_axes(),
                    fill=True,
                    plot_density=True,
                    truth=(
                        [truths[par] for par in parlist] if truths is not None else None
                    ),
                    N=10000,
                    percentiles=[5, 95],
                    rangex=lims[0],
                    rangey=lims[1],
                    grid=True,
                    legend_kwargs={
                        "loc": "upper right",
                        "frameon": True,
                        "framealpha": 0.75,
                        "facecolor": "w",
                        "edgecolor": "w",
                        "fontsize": 12,
                    },
                )
            else:
                if jp >= len(figs):
                    figs.append(
                        figure(
                            figsize=(
                                args.size * (2 + (len(parlist) - 2)),
                                args.size * (2 + (len(parlist) - 2)),
                            ),
                            gca=False,
                        )
                    )
                    gs = gridspec.GridSpec(
                        len(parlist),
                        len(parlist),
                        width_ratios=[1] * len(parlist),
                        height_ratios=[1] * len(parlist),
                        wspace=0.0,
                        hspace=0.0,
                    )
                    axes = [
                        figs[jp].add_subplot(gs[jjj])
                        for jjj in range(len(parlist) ** 2)
                    ]
                    for jj in range(len(parlist)):
                        axes[jj * len(parlist)].minorticks_on()
                        for ii in range(jj + 1):
                            axes[jj * len(parlist) + ii].tick_params(
                                direction="in", which="both", top=False, right=False
                            )
                    axes[0].xaxis.set_ticklabels([])
                ylims_now = [
                    this_ax.get_ylim()[-1]
                    for this_ax in [
                        axes[ii * (len(parlist) + 1)] for ii in range(len(parlist))
                    ]
                ]
                figs[jp], _ = post.make_corner_plot(
                    parlist,
                    lims,
                    color,
                    truths=true_vals,
                    truth_color=truthcolor,
                    fig=figs[jp],
                )
                if args.ticks is not None:
                    for jpar, par in enumerate(parlist):
                        if args.ticks[jp][jpar] is not None:
                            tickvals = eval(args.ticks[jp][jpar])
                            # Diagonal
                            axdiag = figs[jp].get_axes()[jpar * len(parlist) + jpar]
                            axdiag.set_xticks(tickvals)
                            # Off-diagonal x
                            for jrow in range(jpar + 1, len(parlist)):
                                axoffx = figs[jp].get_axes()[jrow * len(parlist) + jpar]
                                axoffx.set_xticks(tickvals)
                            # Off-diagonal y
                            for jcol in range(jpar):
                                axoffy = figs[jp].get_axes()[jpar * len(parlist) + jcol]
                                axoffy.set_yticks(tickvals)
                for ii in range(len(parlist)):
                    axdiag = figs[jp].get_axes()[ii * len(parlist) + ii]
                    axdiag.set_ylim(0, max(ylims_now[ii], axdiag.get_ylim()[-1]))

    if args.label is not None:
        for jf, fig in enumerate(figs):
            if not (args.triangle and len(args.par[jf]) == 2):
                fig.legend(
                    handles=[
                        matplotlib.lines.Line2D([], [], color=col, label=lbl)
                        for col, lbl in zip(line_cols, args.label)
                    ],
                    frameon=False,
                    fontsize=15,
                    loc="upper right",
                )

    for fig, parlist in zip(figs, args.par):
        for any_ax in fig.get_axes():
            any_ax.tick_params(labelsize=9)

    if args.save:
        if args.name is None:
            namestr = ""
            for file in args.input:
                namestr = namestr + file.split("/")[-1].split(".")[-2] + "_"
        else:
            namestr = args.name
        # Use current directory for output by default
        output_dir = args.output_dir if args.output_dir else "."
        for fig, parlist in zip(figs, args.par):
            for artist in fig.get_children():
                try:
                    artist.set_rasterized(True)
                except AttributeError:
                    pass
            parnames = "_".join(parlist)
            fig.savefig(
                os.path.join(output_dir, f"{namestr}{parnames}.png"),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(output_dir, f"{namestr}{parnames}.pdf"),
                bbox_inches="tight",
                dpi=300,
            )
    else:
        plt.show()
