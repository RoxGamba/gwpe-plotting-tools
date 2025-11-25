"""
gwpe-plotting-tools: Plotting tools for GW PE pipelines.

This package provides utilities for plotting gravitational wave
parameter estimation results from bilby and RIFT pipelines.
"""

from .posteriors import Posterior, BilbyPosterior, RIFTPosterior
from .gwutils import compute_chi_prec, compute_chi_prec_from_xyz

__all__ = [
    "Posterior",
    "BilbyPosterior",
    "RIFTPosterior",
    "compute_chi_prec",
    "compute_chi_prec_from_xyz",
]
