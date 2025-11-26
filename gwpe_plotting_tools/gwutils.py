"""
Utilities for computing derived parameters.

This module provides utility functions for computing derived gravitational
wave parameters such as the precessing spin parameter (chi_p).
"""

import numpy as np


def compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2):
    """
    Compute the precessing spin parameter chi_p.

    Computes chi_p from spin magnitudes and tilt angles for a compact
    binary coalescence. The function automatically handles cases where
    m2 > m1 by swapping the masses internally.

    Parameters
    ----------
    m1 : float
        Primary mass component in solar masses.
    m2 : float
        Secondary mass component in solar masses.
    s1 : float
        Primary spin magnitude (dimensionless, 0 to 1).
    s2 : float
        Secondary spin magnitude (dimensionless, 0 to 1).
    tilt1 : float
        Primary spin tilt angle in radians.
    tilt2 : float
        Secondary spin tilt angle in radians.

    Returns
    -------
    float
        The precessing spin parameter chi_p.
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
    Compute chi_p from Cartesian spin components.

    Computes the precessing spin parameter chi_p from the in-plane
    (x, y) spin components. The function handles cases where q < 1
    by inverting the mass ratio and swapping spins internally.

    Parameters
    ----------
    q : float
        Mass ratio (m2/m1 or m1/m2).
    chi1x : float
        Primary spin x-component (dimensionless).
    chi1y : float
        Primary spin y-component (dimensionless).
    chi2x : float
        Secondary spin x-component (dimensionless).
    chi2y : float
        Secondary spin y-component (dimensionless).

    Returns
    -------
    float or ndarray
        The precessing spin parameter chi_p.
    """
    chi1_perp = np.sqrt(chi1x**2 + chi1y**2)
    chi2_perp = np.sqrt(chi2x**2 + chi2y**2)

    if q < 1.0:
        chi1_perp, chi2_perp = chi2_perp, chi1_perp
        q = 1.0 / q

    return np.maximum(chi1_perp, (4.0 + 3.0 * q) / (4.0 * q**2 + 3.0 * q) * chi2_perp)
