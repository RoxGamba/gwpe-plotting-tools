"""
Useful functions for plotting results of PE.
"""

from abc import ABC, abstractmethod
import logging

import corner
import matplotlib.pyplot as plt
import numpy as np
from pesummary.core.plots.publication import _triangle_plot, _triangle_axes
from scipy.stats import gaussian_kde

from .constants import KEYS_LATEX, RIFT_TO_BILBY
from . import gwutils

try:
    from bilby.gw.prior import BBHPriorDict
    from bilby.gw.result import CBCResult
except ImportError:
    logging.warning("bilby not found, bilby-related functions will not work")

# use latex for the labels
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=13)


class Posterior(ABC):
    """
    Abstract base class for posterior sample handling.

    Subclasses must implement the `load` method to load posterior
    samples from their respective file formats.
    """

    def __init__(self, filename):
        """
        Initialize the Posterior.

        Parameters
        ----------
        filename : str
            Path to the posterior samples file.
        """
        self.filename = filename
        self.load(filename)

    @abstractmethod
    def load(self, filename):
        """
        Load posterior samples from file.

        This method must be implemented by subclasses to handle
        loading posterior samples from their specific file formats.
        After loading, posterior parameter samples should be set as
        instance attributes (e.g., self.mass_1, self.chi_eff, etc.).

        Parameters
        ----------
        filename : str
            Path to the posterior samples file.
        """
        pass

    def make_hist(
        self,
        key,
        color,
        fig=None,
        bins=None,
        label=None,
        truth=None,
        percentiles=None,
        **kwargs,
    ):
        data = self.__getattribute__(key)
        if fig is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0]

        if bins is None:
            bins = int(np.sqrt(len(data)))

        ax.hist(
            data,
            density=True,
            histtype="step",
            bins=bins,
            color=color,
            label=label,
            **kwargs,
        )
        if truth is not None:
            ax.axvline(truth, color="k", linestyle="--")
        if percentiles is not None:
            percs = np.percentile(data, percentiles)
            ax.axvline(percs[0], lw=1.25, ls="--", color=color)
            ax.axvline(percs[1], lw=1.25, ls="--", color=color)
        return fig

    def find_maxL(self):
        """
        Find the maximum likelihood sample."""
        logL = self.log_likelihood
        maxL = np.argmax(logL)
        params = {}
        for key in self.__dict__.keys():
            if "prior" not in key:
                params[key] = self.__getattribute__(key)[maxL]
        return params

    def compute_savage_dickey_and_ci(self, keys, gr_values):
        """
        Compute Savage-Dickey ratio and GR quantile,
        adapting KC's script.

        NOTES:
            - here we assume uniform priors on the GR-deviated parameters for the SD ratio calculation.
            - Currently works assuming priors bilby-like, needs to be generalized
        """

        posterior = []
        priors = []
        for this_key in keys:
            posterior.append(np.array(self.__getattribute__(this_key)))
            priors.append(self.__getattribute__(this_key + "-prior"))
        assert len(posterior) == len(gr_values)
        assert len(priors) == len(gr_values)

        gkde = gaussian_kde(np.vstack(posterior))
        posterior_density_gr = gkde(gr_values)
        posterior_density = gkde(np.vstack(posterior))

        # prior density at the gr_point, assume uniform priors!
        den = 1
        for this_prior in priors:
            den *= this_prior.maximum - this_prior.minimum
        prior_density_gr = 1.0 / den

        # SD ratio
        sdratio = prior_density_gr / posterior_density_gr
        log10_sdr = np.log10(sdratio)
        ci_fraction = np.mean(posterior_density >= posterior_density_gr)
        return log10_sdr[0], ci_fraction

    def make_triangle_plot(
        self,
        keys,
        color,
        label,
        fill=True,
        plot_density=False,
        fig=None,
        axes=None,
        N=1000,
        truth=None,
        percentiles=None,
        figsize=(5, 5),
        grid=False,
        **kde_args,
    ):
        """
        Make a triangle plot using pesummary
        Example usage:
            pst.make_triangle_plot(["mass_1", "mass_2"], "r", "Test", N=3000, plot_density=False, fill=False)
        """
        assert len(keys) == 2

        if fig is None:
            fig, ax1, temp, ax2, ax3 = _triangle_axes(figsize=figsize)
            temp.remove()
        else:
            ax1, ax2, ax3 = axes

        rand_idx = np.random.choice(range(len(self.__getattribute__(keys[0]))), N)
        posts = [np.array(self.__getattribute__(key))[rand_idx] for key in keys]
        labels = [KEYS_LATEX[key] for key in keys]
        fig, ax1, ax2, ax3 = _triangle_plot(
            x=posts[0],
            y=posts[1],
            xlabel=labels[0],
            ylabel=labels[1],
            colors=[color],
            labels=[label],
            fill=fill,
            fill_alpha=0.25,
            plot_density=plot_density,
            fig=fig,
            axes=(ax1, ax2, ax3),
            # truth=truth,
            **kde_args,
        )
        # Manually add percentile lines to have them in the right color
        if percentiles is not None:
            assert len(percentiles) == 2
            percs = np.percentile(posts[0], percentiles)
            ax1.axvline(percs[0], lw=1.25, ls="--", color=color)
            ax1.axvline(percs[1], lw=1.25, ls="--", color=color)
            percs = np.percentile(posts[1], percentiles)
            ax3.axhline(percs[0], lw=1.25, ls="--", color=color)
            ax3.axhline(percs[1], lw=1.25, ls="--", color=color)
        # Also manually add truth lines
        if truth is not None:
            ax1.axvline(truth[0], lw=0.5, ls="-.", color="k")
            ax3.axhline(truth[1], lw=0.5, ls="-.", color="k")
            ax2.axvline(truth[0], lw=0.5, ls="-.", color="k")
            ax2.axhline(truth[1], lw=0.5, ls="-.", color="k")
            ax2.scatter(truth[0], truth[1], marker="s", s=25, c="k", zorder=10)
        if grid:
            ax2.grid(alpha=0.75, ls=":")

        return fig, [ax1, ax2, ax3]

    def make_corner_plot(
        self,
        keys,
        limits,
        color,
        ylimits=None,
        fig=None,
        bin=50,
        lbl=None,
        plot_maxL=False,
        truths=None,
        truth_color=None,
    ):
        """
        Make a corner plot for the posterior samples.
        keys: list of keys to plot
        range: list of ranges for each key
        color: color of the plot
        fig: figure to plot on (optional)
        bin: number of bins (optional)
        lbl: label for the histogram (optional)
        """

        matrix = np.transpose([self.__getattribute__(key) for key in keys])
        labels = [KEYS_LATEX[key] for key in keys]

        fig, axes = make_corner_plot(
            matrix,
            labels,
            limits,
            color,
            ylimits=ylimits,
            fig=fig,
            bin=bin,
            lbl=lbl,
            truths=truths,
            truth_color=truth_color,
        )

        if plot_maxL:
            maxL_pars = self.find_maxL()
            for i, key in enumerate(keys):
                if key in maxL_pars.keys():
                    ax = axes[i, i]
                    ax.axvline(
                        maxL_pars[key], color=color, linestyle="-", linewidth=1.5
                    )

            # add stars for the maxL points to 2D histograms
            for i in range(len(keys)):
                for j in range(i):
                    ax = axes[i, j]
                    if keys[i] in maxL_pars.keys() and keys[j] in maxL_pars.keys():
                        ax.plot(
                            maxL_pars[keys[j]], maxL_pars[keys[i]], "*", color=color
                        )

        return fig, axes


class BilbyPosterior(Posterior):
    """
    Posterior class for bilby result files.

    Supports loading from JSON, HDF5, and pickle file formats.
    """

    def load(self, filename):
        """
        Load posterior samples from a bilby result file.

        Parameters
        ----------
        filename : str
            Path to the bilby result file (.json, .hdf5, or .pkl)
        """
        if "json" in filename:
            result = CBCResult.from_json(filename)
        elif "hdf5" in filename:
            result = CBCResult.from_hdf5(filename)
        elif "pkl" in filename:
            result = CBCResult.from_pkl(filename)
        else:
            raise ValueError("Unknown bilby file type")

        posterior = result.posterior
        prior = result.priors

        self.bilby_result = result
        self.log_bayes_factor = result.log_bayes_factor

        for this_key in posterior.keys():
            try:
                self.__setattr__(this_key, posterior[this_key])
            except (KeyError, AttributeError) as e:
                logging.debug("Could not set posterior key %s: %s", this_key, e)

        for this_key in prior.keys():
            try:
                self.__setattr__(this_key + "-prior", prior[this_key])
            except (KeyError, AttributeError) as e:
                logging.debug("Could not set prior key %s: %s", this_key, e)

    def reconstruct_waveforms(self, ifos=None, save=False):
        """
        Reconstruct waveforms from the posterior samples.

        Parameters
        ----------
        ifos : list, optional
            List of interferometers to reconstruct waveforms for.
            If None, will use all available interferometers in the bilby result.
        save : bool, optional
            Whether to save the waveform plots. Default is False.
        """

        if ifos is None:
            try:
                ifos = self.bilby_result.interferometers
            except AttributeError:
                logging.warning(
                    "No interferometers found in bilby result; we will not plot the data and"
                    " will assume H1, L1, V1 for waveform reconstruction."
                )
                ifos = ["H1", "L1", "V1"]

        figs = {}
        for ifo in ifos:
            ifo_fig = self.bilby_result.plot_interferometer_waveform_posterior(
                interferometer=ifo,
                n_samples=500,
                save=save,
            )
            figs[ifo] = ifo_fig

        return figs

    def draw_from_prior(self, n_samples=1000):
        """
        Draw samples from the prior.
        n_samples: number of samples to draw
        """
        raise NotImplementedError("Bilby prior sampling not implemented yet.")


class RIFTPosterior(Posterior):
    """
    Posterior class for RIFT result files.

    Loads posterior samples from RIFT extrinsic_posterior_samples.dat files.
    """

    def load(self, filename):
        """
        Load posterior samples from a RIFT result file.

        Parameters
        ----------
        filename : str
            Path to the RIFT posterior samples file
        """
        with open(filename, "r") as f:
            data = np.genfromtxt(f, names=True)

        for name in data.dtype.names:
            if name in RIFT_TO_BILBY:
                self.__setattr__(RIFT_TO_BILBY[name], data[name])
            else:
                logging.warning("Unknown RIFT parameter: %s", name)

    def reconstruct_waveforms(self):
        """
        Reconstruct waveforms from the posterior samples.
        """
        raise NotImplementedError("RIFT waveform reconstruction not implemented yet.")

    def draw_from_prior(self, n_samples=1000):
        """
        Draw samples from the prior.
        n_samples: number of samples to draw
        """
        raise NotImplementedError("RIFT prior sampling not implemented yet.")


def make_corner_plot(
    matrix,
    labels,
    limits,
    color,
    ylimits=None,
    fig=None,
    bin=30,
    lbl=None,
    truths=None,
    truth_color=None,
):

    L = max(len(matrix[0]), len(np.transpose(matrix)[0]))
    N = int(min(len(matrix[0]), len(np.transpose(matrix)[0])))

    if truth_color is None:
        truth_color = color

    if fig is None:
        fig = cornerfig = corner.corner(
            matrix,
            labels=labels,
            weights=np.ones(L) * 100.0 / L,
            bins=bin,
            range=limits,
            color=color,
            levels=[0.5, 0.9],
            quantiles=[0.05, 0.95],
            contour_kwargs={"colors": color, "linewidths": 0.95},
            label_kwargs={"size": 15.0},
            hist2d_kwargs={"label": lbl},
            # hist_kwargs     = {'density':True},
            plot_datapoints=False,
            show_titles=False,
            plot_density=True,
            smooth1d=True,
            smooth=True,
            truths=truths,
            truth_color=truth_color,
            labelpad=0.085,
        )
    else:
        fig = cornerfig = corner.corner(
            matrix,
            fig=fig,
            weights=np.ones(L) * 100.0 / L,
            labels=labels,
            range=limits,
            bins=bin,
            color=color,
            levels=[0.5, 0.9],
            quantiles=[0.05, 0.95],
            contour_kwargs={"colors": color, "linewidths": 0.95},
            label_kwargs={"size": 15.0},
            hist2d_kwargs={"label": lbl},
            # hist_kwargs     = {'density':True},
            plot_datapoints=False,
            show_titles=False,
            plot_density=True,
            smooth1d=True,
            smooth=True,
            truths=truths,
            truth_color=truth_color,
            labelpad=0.085,
        )
    axes = np.array(cornerfig.axes).reshape((N, N))

    if ylimits is not None:
        for i in np.arange(N):
            if ylimits[i] is None:
                continue
            ax = axes[i, i]
            ax.set_ylim((0, ylimits[i]))

    return fig, axes


def prior_from_file(fname):
    prior_dict = BBHPriorDict({})
    prior_dict.from_file(fname)
    return prior_dict


def sample_from_prior(fname, n_samples=1000):
    prior_dict = prior_from_file(fname)
    samples = prior_dict.sample(n_samples)

    samples["mass_1"] = (
        samples["chirp_mass"]
        * (1 + samples["mass_ratio"]) ** (1 / 5)
        / samples["mass_ratio"] ** (3 / 5)
    )
    samples["mass_2"] = samples["mass_1"] * samples["mass_ratio"]

    samples["chi_p"] = np.vectorize(gwutils.compute_chi_prec)(
        samples["mass_1"],
        samples["mass_2"],
        samples["a_1"],
        samples["a_2"],
        samples["tilt_1"],
        samples["tilt_2"],
    )

    samples["chi_eff"] = (
        samples["a_1"] * np.cos(samples["tilt_1"])
        + samples["a_2"] * np.cos(samples["tilt_2"]) * samples["mass_ratio"]
    ) / (1 + samples["mass_ratio"])

    return samples


def create_posterior(filename, kind):
    """
    Factory function to create the appropriate Posterior subclass.

    Parameters
    ----------
    filename : str
        Path to the posterior samples file
    kind : str
        Type of file: 'bilby-json', 'bilby-hdf5', 'bilby-pkl', or 'rift'

    Returns
    -------
    Posterior
        An instance of BilbyPosterior or RIFTPosterior
    """
    if kind in ["bilby-json", "bilby-hdf5", "bilby-pkl"]:
        return BilbyPosterior(filename)
    elif kind == "rift":
        return RIFTPosterior(filename)
    else:
        raise ValueError(f"Unknown file kind: {kind}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="Input file name")
    parser.add_argument(
        "--kind",
        type=str,
        default="bilby-json",
        help="File kind: bilby-json, bilby-hdf5, bilby-pkl, rift",
    )
    parser.add_argument("--key1", type=str, default="mass_1", help="First key to plot")
    parser.add_argument("--key2", type=str, default="mass_2", help="Second key to plot")
    parser.add_argument(
        "--test-triangle", action="store_true", help="Test triangle plot"
    )
    parser.add_argument("--test-corner", action="store_true", help="Test corner plot")
    parser.add_argument(
        "--test-sd", action="store_true", help="Test Savage-Dickey calculation"
    )
    args = parser.parse_args()

    if args.kind not in ["bilby-json", "bilby-hdf5", "bilby-pkl", "rift"]:
        raise ValueError("Unknown file kind")

    if not (args.test_triangle or args.test_corner or args.test_sd):
        raise ValueError(
            "Please specify at least one test to run: "
            "--test-triangle, --test-corner, --test-sd"
        )

    post = create_posterior(args.fname, args.kind)
    if args.test_corner:
        lim_key1 = [
            min(post.__getattribute__(args.key1)),
            max(post.__getattribute__(args.key1)),
        ]
        lim_key2 = [
            min(post.__getattribute__(args.key2)),
            max(post.__getattribute__(args.key2)),
        ]
        post.make_corner_plot(
            [args.key1, args.key2],
            limits=[lim_key1, lim_key2],
            color="r",
            plot_maxL=True,
        )
        plt.show()
    if args.test_triangle:
        post.make_triangle_plot(
            [args.key1, args.key2],
            color="r",
            label="Test",
            N=3000,
            plot_density=False,
            fill=False,
        )
        plt.show()
    if args.test_sd:
        print(post.compute_savage_dickey_and_ci([args.key1, args.key2], [0, 0]))
