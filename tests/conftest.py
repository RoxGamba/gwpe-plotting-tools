"""
Pytest configuration and fixtures for gwpe_plotting_tools tests.
"""

import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def sample_rift_posterior_data():
    """Create sample RIFT-format posterior data.
    
    Returns a dict with keys matching RIFT parameter names and numpy arrays
    representing posterior samples.
    """
    np.random.seed(42)
    n_samples = 100
    
    # Generate mock parameter samples
    data = {
        "m1": 30.0 + np.random.randn(n_samples) * 2.0,
        "m2": 25.0 + np.random.randn(n_samples) * 1.5,
        "a1x": np.random.randn(n_samples) * 0.1,
        "a1y": np.random.randn(n_samples) * 0.1,
        "a1z": 0.3 + np.random.randn(n_samples) * 0.1,
        "a2x": np.random.randn(n_samples) * 0.1,
        "a2y": np.random.randn(n_samples) * 0.1,
        "a2z": 0.2 + np.random.randn(n_samples) * 0.1,
        "ra": np.random.uniform(0, 2 * np.pi, n_samples),
        "dec": np.random.uniform(-np.pi / 2, np.pi / 2, n_samples),
        "distance": 500.0 + np.random.randn(n_samples) * 50.0,
        "incl": np.random.uniform(0, np.pi, n_samples),
        "psi": np.random.uniform(0, np.pi, n_samples),
        "phiorb": np.random.uniform(0, 2 * np.pi, n_samples),
        "time": 1187008882.0 + np.random.randn(n_samples) * 0.001,
        "lnL": -50.0 + np.random.randn(n_samples) * 5.0,
    }
    
    # Ensure m1 > m2
    mask = data["m1"] < data["m2"]
    data["m1"][mask], data["m2"][mask] = data["m2"][mask].copy(), data["m1"][mask].copy()
    
    return data


@pytest.fixture
def sample_rift_file(sample_rift_posterior_data, tmp_path):
    """Create a sample RIFT posterior file for testing.
    
    Returns the path to a temporary file containing RIFT-format posterior data.
    """
    data = sample_rift_posterior_data
    
    # Create the header line
    header = " ".join(data.keys())
    
    # Create the data matrix
    matrix = np.column_stack([data[key] for key in data.keys()])
    
    # Write to file
    filepath = tmp_path / "test_rift_posterior.dat"
    np.savetxt(filepath, matrix, header=header, comments="")
    
    return str(filepath)


@pytest.fixture
def mock_masses():
    """Return typical binary black hole mass parameters."""
    return {
        "m1": 30.0,
        "m2": 25.0,
        "mass_ratio": 25.0 / 30.0,
    }


@pytest.fixture
def mock_spins():
    """Return typical binary black hole spin parameters."""
    return {
        "s1": 0.5,
        "s2": 0.3,
        "tilt1": 0.3,  # radians
        "tilt2": 0.5,  # radians
        "chi1x": 0.1,
        "chi1y": 0.2,
        "chi2x": 0.05,
        "chi2y": 0.15,
    }


@pytest.fixture(scope="session")
def matplotlib_backend():
    """Set up non-interactive matplotlib backend for testing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
