"""
Utilities for computing derived parameters.
"""

import numpy as np


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
