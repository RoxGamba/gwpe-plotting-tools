"""
Constants and mappings for gwpe-plotting-tools.

This module contains parameter name mappings and LaTeX labels
used throughout the package.
"""

# Mapping from RIFT parameter names to bilby parameter names
RIFT_TO_BILBY = {
    "m1": "mass_1",
    "m2": "mass_2",
    "a1x": "spin_1x",
    "a1y": "spin_1y",
    "a1z": "spin_1z",
    "a2x": "spin_2x",
    "a2y": "spin_2y",
    "a2z": "spin_2z",
    "mc": "chirp_mass",
    "eta": "symmetric_mass_ratio",
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

BAJES_TO_BILBY = {
    "mchirp": "chirp_mass",
    "q": "mass_ratio",
    "s1x": "spin_1x",
    "s1y": "spin_1y",
    "s1z": "spin_1z",
    "s2x": "spin_2x",
    "s2y": "spin_2y",
    "s2z": "spin_2z",
    "lambda1": "lambda_1",  # double check
    "lambda2": "lambda_2",  # double check
    "energy": "energy",  # to be confirmed with rift
    "angmom": "angular_momentum",  # to be confirmed with rift
    "cosi": "cos_iota",
    "phi_ref": "phase",
    "psi": "psi",
    "ra": "ra",
    "dec": "dec",
    "distance": "luminosity_distance",
    "time_shift": "time_shift",  # Not sure this has a bilby equivalent!
    "logL": "log_likelihood",
}

# Extended mapping including additional parameters from MDC files
MDC_TO_BILBY = {
    **RIFT_TO_BILBY,
    "tref": "geocent_time",
    "dist": "luminosity_distance",
    "total_mass": "total_mass",
    "mass_ratio": "mass_ratio",
    "indx": "indx",
    "delta_taulm0": "delta_taulm0",
    "delta_omglm0": "delta_omglm0",
    "iota": "iota",
    "delta_Mbhf": "delta_Mbhf",
    "delta_abhf": "delta_abhf",
}

# LaTeX labels for parameters
KEYS_LATEX = {
    "mass_ratio": r"$1/q$",
    "chirp_mass": r"$\mathcal{M}_c [M_{\odot}]$",
    "total_mass": r"$M [M_{\odot}]$",
    "mass_1": r"$m_1 [M_{\odot}]$",
    "mass_2": r"$m_2 [M_{\odot}]$",
    "symmetric_mass_ratio": r"$\nu$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "luminosity_distance": r"$d_L$ [Mpc]",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
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
    "delta_alphalm0": r"$\delta \alpha_{220}$",
    "delta_taulm0": r"$\delta \tau_{220}$",
    "delta_omglm0": r"$\delta \omega_{220}$",
    "delta_abhf": r"$\delta a_{\rm BH}^f$",
    "delta_Mbhf": r"$\delta M_{\rm BH}^f$",
    "delta_a6c": r"$\delta a_6^c$",
    "delta_cN3LO": r"$\delta c_{\rm{N}^3\rm{LO}}$",
    "geocent_time": r"$t_c$",
    "log_likelihood": r"$\log \mathcal{L}$",
}
