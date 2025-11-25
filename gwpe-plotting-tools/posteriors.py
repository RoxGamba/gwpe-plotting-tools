"""
Useful functions for plotting results of PE.

Adapted from RG's script.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
sys.path.append('./figure_scripts')
import figure_scripts.corner as corner
# from pesummary.core.plots.publication import _triangle_plot, _triangle_axes
# Use custom modified version of pesummary.publication for greater control of aesthetical details
from figure_scripts.publication import _triangle_plot, _triangle_axes
import json
from scipy.stats import gaussian_kde

import seaborn as sns
from bilby.gw.prior import BBHPriorDict
from bilby.gw.result import CBCResult

# set colormap to colorblind

# Colors for scripts
p_green  = (0.0, 0.6078431372549019, 0.6196078431372549)
p_pink   = (0.7803921568627451, 0.36470588235294116, 0.6705882352941176)
p_purple = (0.2, 0.13333333333333333, 0.5333333333333333)

# use latex for the labels
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=13)


def compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2):
    """Compute chi precessing spin parameter (for given 3-dim spin vectors)
    --------
    m1 = primary mass component [solar masses]
    m2 = secondary mass component [solar masses]
    s1 = primary spin megnitude [dimensionless]
    s2 = secondary spin megnitude [dimensionless]
    tilt1 = primary spin tilt [rad]
    tilt2 = secondary spin tilt [rad]
    """

    s1_perp = np.abs(s1 * np.sin(tilt1))
    s2_perp = np.abs(s2 * np.sin(tilt2))
    one_q = m2 / m1

    # check that m1>=m2, otherwise switch
    if one_q > 1.0:
        one_q = 1.0 / one_q
        s1_perp, s2_perp = s2_perp, s1_perp

    return np.max(
        [s1_perp, s2_perp * one_q * (4.0 * one_q + 3.0) / (3.0 * one_q + 4.0)]
    )


def compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y):
    """
    Compute chi_p given spin components
    """

    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)

    if q < 1.0:
        chi1_perp, chi2_perp = chi2_perp, chi1_perp
        q = 1.0 / q

    return np.maximum(chi1_perp, (4.0 + 3.0 * q) / (4.0 * q**2 + 3.0 * q) * chi2_perp)


class Posterior(object):
    def __init__(self, filename, kind):

        if "bilby" in kind:
            self.load_bilby(filename, kind)
        elif kind == "o3a-hdf5":
            self.load_o3a_hdf5(filename)
        elif kind == "gwtc1":
            self.load_gwtc1(filename)
        elif kind == "rift":
            self.load_rift(filename)
        elif kind == "txt":
            self.load_txt(filename)
        else:
            raise ValueError("Unknown file type")

    def load_gwtc1(self, filename):
        data_gwtc = h5py.File(filename, "r")
        m1 = data_gwtc["Overall_posterior"]["m1_detector_frame_Msun"][:]
        m2 = data_gwtc["Overall_posterior"]["m2_detector_frame_Msun"][:]
        Dl = data_gwtc["Overall_posterior"]["luminosity_distance_Mpc"][:]
        M = m1 + m2
        q = m2 / m1
        Mc = M * (q / (1 + q) / (1 + q)) ** (3.0 / 5.0)
        s1 = data_gwtc["Overall_posterior"]["spin1"][:]
        s2 = data_gwtc["Overall_posterior"]["spin2"][:]
        ct1 = data_gwtc["Overall_posterior"]["costilt1"][:]
        ct2 = data_gwtc["Overall_posterior"]["costilt2"][:]
        tilt1 = np.arccos(ct1)
        tilt2 = np.arccos(ct2)
        chi1z = s1 * ct1
        chi2z = s2 * ct2

        # compute chip and
        chip = [
            compute_chi_prec(m1i, m2i, s1i, s2i, tilt1i, tilt2i)
            for m1i, m2i, s1i, s2i, tilt1i, tilt2i in zip(m1, m2, s1, s2, tilt1, tilt2)
        ]
        chip = np.array(chip)
        chi_eff = np.array((m1 * chi1z + m2 * chi2z) / M)

        self.__setattr__(
            "cos_theta_jn", data_gwtc["Overall_posterior"]["costheta_jn"][:]
        )
        self.__setattr__("mass_ratio", q)
        self.__setattr__("chirp_mass", Mc)
        self.__setattr__("a_1", s1)
        self.__setattr__("a_2", s2)
        self.__setattr__("chi_eff", chi_eff)
        self.__setattr__("chi_p", chip)
        self.__setattr__("luminosity_distance", Dl)

    def load_o3a_hdf5(self, filename):
        data = h5py.File(filename, "r")
        default_keys = [
            "mass_ratio",
            "chirp_mass",
            "a_1",
            "a_2",
            "luminosity_distance",
            "iota",
            "cos_theta_jn",
            "chi_eff",
            "chi_1",
            "chi_2",
            "chi_p",
            "log_likelihood",
        ]

        for this_key in default_keys:
            try:
                self.__setattr__(
                    this_key, data["ProdF4"]["posterior_samples"][this_key][:]
                )
            except Exception:
                continue
                # print(f"Key {this_key} not found in {filename}")

    def load_rift(self, filename):
        with open(filename, "r") as f:
            data = np.genfromtxt(f, names=True)

        # m1 m2 a1x a1y a1z a2x a2y a2z mc eta  ra dec time phiorb incl psi
        # distance Npts lnL p ps neff  mtotal q chi_eff chi_p  m1_source m2_source mc_source
        # mtotal_source redshift  eccentricity meanPerAno
        # rift to bilby
        rift_to_bilby = {
            "m1": "mass_1",
            "m2": "mass_2",
            "a1x": "spin_1x",
            "a1y": "spin_1y",
            "a1z": "spin_1z",
            "a2x": "spin_2x",
            "a2y": "spin_2y",
            "a2z": "spin_2z",
            "mc": "chirp_mass",
            "eta": "symmetric_mass_ratio",  # CHECKME
            "ra": "ra",
            "dec": "dec",
            "time": "geocent_time",
            "phiorb": "phase",
            "incl": "iota",
            "psi": "psi",
            "distance": "luminosity_distance",
            "Npts": "npts",
            "lnL": "log_likelihood",
            "p": "p",
            "ps": "ps",
            "neff": "neff",
            "mtotal": "total_mass",
            "q": "mass_ratio",
            "chi_eff": "chi_eff",
            "chi_p": "chi_p",
            "m1_source": "mass_1_source",
            "m2_source": "mass_2_source",
            "mc_source": "chirp_mass_source",
            "mtotal_source": "total_mass_source",
            "redshift": "redshift",
            "eccentricity": "eccentricity",
            "meanPerAno": "mean_per_ano",
        }

        for name in data.dtype.names:
            try:
                self.__setattr__(rift_to_bilby[name], data[name])
            except Exception:
                print("Problem with key: ", name)
                continue
                # print(f"Key {name} not found in {filename}")

    def load_bilby(self, filename, kind):
        if "json" in kind:
            result = CBCResult.from_json(filename)
        elif "hdf5" in kind:
            result = CBCResult.from_hdf5(filename)
        elif "pkl" in kind:
            result = CBCResult.from_pkl(filename)
        else:
            raise ValueError("Unknown bilby file type")

        posterior = result.posterior
        prior = result.priors

        self.log_bayes_factor = result.log_bayes_factor

        for this_key in posterior.keys():
            try:
                self.__setattr__(this_key, posterior[this_key])
            except Exception:
                continue
                # print(f"Key {this_key} not found in {filename}")

        for this_key in prior.keys():
            try:
                self.__setattr__(this_key + "-prior", prior[this_key])
            except Exception:
                continue

        pass

    def load_txt(self, filename):
        data = np.genfromtxt(filename, names=True, dtype=float)
        for key in data.dtype.names:
            self.__setattr__(key, data[key])
        pass

    def make_hist(self, key, color, fig=None, bins=None, label=None, truth=None, percentiles=None, **kwargs):
        data = self.__getattribute__(key)
        if fig is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0]

        if bins is None:
            bins = int(np.sqrt(len(data)))

        ax.hist(data, density=True, histtype="step", bins=bins, color=color, label=label, **kwargs)
        if truth is not None:
            ax.axvline(truth, color='k', linestyle='--')
        if percentiles is not None:
            percs = np.percentile(data, percentiles)
            ax.axvline(percs[0], lw=1.25, ls='--', color=color)
            ax.axvline(percs[1], lw=1.25, ls='--', color=color)
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
        figsize=(5,5),
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
        labels = [keys_latex[key] for key in keys]
        fig, ax1, ax2, ax3 = _triangle_plot(
            x=posts[0],
            y=posts[1],
            xlabel=labels[0],
            ylabel=labels[1],
            colors=[color],
            labels=[label],
            fill=fill, fill_alpha=0.25,
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
            ax1.axvline(percs[0], lw=1.25, ls='--', color=color)
            ax1.axvline(percs[1], lw=1.25, ls='--', color=color)
            percs = np.percentile(posts[1], percentiles)
            ax3.axhline(percs[0], lw=1.25, ls='--', color=color)
            ax3.axhline(percs[1], lw=1.25, ls='--', color=color)
        # Also manually add truth lines
        if truth is not None:
            ax1.axvline(truth[0], lw=0.5, ls='-.', color='k')
            ax3.axhline(truth[1], lw=0.5, ls='-.', color='k')
            ax2.axvline(truth[0], lw=0.5, ls='-.', color='k')
            ax2.axhline(truth[1], lw=0.5, ls='-.', color='k')
            ax2.scatter(truth[0], truth[1], marker='s', s=25, c='k', zorder=10)
        if grid:
            ax2.grid(alpha=0.75, ls=':')

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
        labels = [keys_latex[key] for key in keys]

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

    if fig == None:
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

    samples["chi_p"] = np.vectorize(compute_chi_prec)(
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


keys_latex = {
    "mass_ratio": r"$1/q$",
    "chirp_mass": r"$\mathcal{M}_c [M_{\odot}]$",
    "total_mass": r"$M [M_{\odot}]$",
    "mass_1": r"$m_1 [M_{\odot}]$",
    "mass_2": r"$m_2 [M_{\odot}]$",
    "symmetric_mass_ratio": r"$\nu$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "luminosity_distance": r"$d_L$ [Mpc]",
    "iota": r"$\iota$ [rad]",
    "dec": r"$\delta [rad]$",
    "ra": r"$\alpha [rad]$",
    "psi": r"$\psi [rad]$",
    "phase": r"$\phi [rad]$",
    "chi_eff": r"$\chi_{\rm eff}$",
    "chi_p": r"$\chi_{\rm p}$",
    "eccentricity": r"$e_{\rm 20 Hz}$",
    "mean_per_ano": r"$\zeta_{\rm 20 Hz}$ [rad]",
    "eccentricity_gw": r"$e_{\rm 20 Hz}^{\rm GW}$",
    "mean_per_ano_gw": r"$\zeta_{\rm 20 Hz}^{\rm GW}$ [rad]",
    "theta_jn": r"$\theta_{\rm JN}$",
    "cos_theta_jn": r"$\cos(\theta_{\rm JN})$",
    # "delta_alphalm0": r"$\delta \alpha_{\ell m 0}$",
    "delta_alphalm0": r"$\delta \alpha_{220}$",
    # "delta_taulm0": r"$\delta \tau_{\ell m 0}$",
    # "delta_omglm0": r"$\delta \omega_{\ell m 0}$",
    "delta_taulm0": r"$\delta \tau_{220}$",
    "delta_omglm0": r"$\delta \omega_{220}$",
    "delta_abhf": r"$\delta a_{\rm BH}^f$",
    "delta_Mbhf": r"$\delta M_{\rm BH}^f$",
    "delta_a6c": r"$\delta a_6^c$",
    "delta_cN3LO": r"$\delta c_{\rm{N}^3\rm{LO}}$",
    "geocent_time": r'$t_c$',
    "log_likelihood": r'$\log \mathcal{L}$',
}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="Input file name")
    parser.add_argument(
        "--kind",
        type=str,
        default="bilby-json",
        help="File kind: bilby-json, bilby-hdf5, bilby-pkl, o3a-hdf5, gwtc1, rift",
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

    if args.kind not in [
        "bilby-json",
        "bilby-hdf5",
        "bilby-pkl",
        "o3a-hdf5",
        "gwtc1",
        "rift",
    ]:
        raise ValueError("Unknown file kind")

    if not (args.test_triangle or args.test_corner or args.test_sd):
        raise ValueError(
            "Please specify at least one test to run: --test-triangle, --test-corner, --test-sd"
        )

    post = Posterior(args.fname, args.kind)
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
