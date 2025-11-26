"""
Tests for gwpe_plotting_tools.posteriors module.

Tests the Posterior classes for loading and handling posterior samples.
"""

import numpy as np
import pytest

from gwpe_plotting_tools.posteriors import (
    Posterior,
    BilbyPosterior,
    RIFTPosterior,
    create_posterior,
)


class TestRIFTPosterior:
    """Tests for RIFTPosterior class."""

    def test_load_from_file(self, sample_rift_file):
        """Test loading posterior from RIFT file."""
        posterior = RIFTPosterior(sample_rift_file)

        # Should have filename attribute
        assert posterior.filename == sample_rift_file

    def test_parameters_renamed(self, sample_rift_file):
        """Test that parameters are renamed from RIFT to bilby convention."""
        posterior = RIFTPosterior(sample_rift_file)

        # Check renamed parameters exist
        assert hasattr(posterior, "mass_1")
        assert hasattr(posterior, "mass_2")
        assert hasattr(posterior, "spin_1z")
        assert hasattr(posterior, "luminosity_distance")
        assert hasattr(posterior, "log_likelihood")

    def test_mass_values_reasonable(self, sample_rift_file):
        """Test that loaded mass values are reasonable."""
        posterior = RIFTPosterior(sample_rift_file)

        # Check mass values are positive and in expected range
        # Allow some tolerance for m1 >= m2 due to random sampling
        mass_tolerance = 5.0  # tolerance in solar masses
        assert np.all(posterior.mass_1 > 0)
        assert np.all(posterior.mass_2 > 0)
        assert np.all(posterior.mass_1 >= posterior.mass_2 - mass_tolerance)

    def test_sample_count(self, sample_rift_file, sample_rift_posterior_data):
        """Test that correct number of samples are loaded."""
        posterior = RIFTPosterior(sample_rift_file)
        expected_count = len(sample_rift_posterior_data["m1"])

        assert len(posterior.mass_1) == expected_count

    def test_reconstruct_waveforms_not_implemented(self, sample_rift_file):
        """Test that waveform reconstruction raises NotImplementedError."""
        posterior = RIFTPosterior(sample_rift_file)

        with pytest.raises(NotImplementedError):
            posterior.reconstruct_waveforms()

    def test_draw_from_prior_not_implemented(self, sample_rift_file):
        """Test that prior sampling raises NotImplementedError."""
        posterior = RIFTPosterior(sample_rift_file)

        with pytest.raises(NotImplementedError):
            posterior.draw_from_prior()


class TestPosteriorBase:
    """Tests for Posterior base class methods using RIFTPosterior."""

    def test_find_maxL(self, sample_rift_file):
        """Test finding maximum likelihood sample."""
        posterior = RIFTPosterior(sample_rift_file)
        maxL_params = posterior.find_maxL()

        # Should return a dictionary
        assert isinstance(maxL_params, dict)

        # Should contain mass parameters
        assert "mass_1" in maxL_params
        assert "mass_2" in maxL_params

        # Should contain a scalar value
        assert isinstance(maxL_params["mass_1"], (float, np.floating))

    def test_find_maxL_corresponds_to_max(self, sample_rift_file):
        """Test that maxL sample corresponds to actual maximum likelihood."""
        posterior = RIFTPosterior(sample_rift_file)
        maxL_params = posterior.find_maxL()

        # Find index of max log_likelihood manually
        max_idx = np.argmax(posterior.log_likelihood)

        # Check that the mass values match
        assert maxL_params["mass_1"] == posterior.mass_1[max_idx]
        assert maxL_params["mass_2"] == posterior.mass_2[max_idx]

    def test_make_hist(self, sample_rift_file, matplotlib_backend):
        """Test histogram creation."""
        posterior = RIFTPosterior(sample_rift_file)
        fig = posterior.make_hist("mass_1", color="blue")

        # Should return a figure
        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)

    def test_make_hist_with_options(self, sample_rift_file, matplotlib_backend):
        """Test histogram creation with various options."""
        posterior = RIFTPosterior(sample_rift_file)
        fig = posterior.make_hist(
            "mass_1",
            color="red",
            bins=20,
            label="Test",
            truth=30.0,
            percentiles=[5, 95],
        )

        # Should return a figure
        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)


class TestCreatePosterior:
    """Tests for create_posterior factory function."""

    def test_create_rift_posterior(self, sample_rift_file):
        """Test creating RIFT posterior via factory function."""
        posterior = create_posterior(sample_rift_file, "rift")

        assert isinstance(posterior, RIFTPosterior)

    def test_create_unknown_kind_raises(self, sample_rift_file):
        """Test that unknown file kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown file kind"):
            create_posterior(sample_rift_file, "unknown")


class TestPosteriorAbstract:
    """Tests for Posterior abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that Posterior cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Posterior("dummy.dat")


class TestCornerCompatibility:
    """Tests for corner plot compatibility."""

    def test_make_corner_plot(self, sample_rift_file, matplotlib_backend):
        """Test corner plot creation with mock data."""
        posterior = RIFTPosterior(sample_rift_file)

        # Define plot parameters
        keys = ["mass_1", "mass_2"]
        lim_mass1 = [
            float(np.min(posterior.mass_1)),
            float(np.max(posterior.mass_1)),
        ]
        lim_mass2 = [
            float(np.min(posterior.mass_2)),
            float(np.max(posterior.mass_2)),
        ]

        fig, axes = posterior.make_corner_plot(
            keys,
            limits=[lim_mass1, lim_mass2],
            color="blue",
        )

        # Should return figure and axes
        assert fig is not None
        assert axes is not None

        # Clean up
        matplotlib_backend.close(fig)

    def test_make_corner_plot_with_truths(self, sample_rift_file, matplotlib_backend):
        """Test corner plot with truth values."""
        posterior = RIFTPosterior(sample_rift_file)

        keys = ["mass_1", "mass_2"]
        lim_mass1 = [
            float(np.min(posterior.mass_1)),
            float(np.max(posterior.mass_1)),
        ]
        lim_mass2 = [
            float(np.min(posterior.mass_2)),
            float(np.max(posterior.mass_2)),
        ]

        fig, axes = posterior.make_corner_plot(
            keys,
            limits=[lim_mass1, lim_mass2],
            color="red",
            truths=[30.0, 25.0],
        )

        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)


class TestPesummaryCompatibility:
    """Tests for pesummary triangle plot compatibility."""

    def test_make_triangle_plot(self, sample_rift_file, matplotlib_backend):
        """Test triangle plot creation with mock data."""
        posterior = RIFTPosterior(sample_rift_file)

        keys = ["mass_1", "mass_2"]

        fig, axes = posterior.make_triangle_plot(
            keys,
            color="blue",
            label="Test",
            N=50,  # Small N for faster test
        )

        # Should return figure and axes
        assert fig is not None
        assert axes is not None
        assert len(axes) == 3  # ax1, ax2, ax3

        # Clean up
        matplotlib_backend.close(fig)

    def test_make_triangle_plot_with_options(
        self, sample_rift_file, matplotlib_backend
    ):
        """Test triangle plot with various options."""
        posterior = RIFTPosterior(sample_rift_file)

        keys = ["mass_1", "mass_2"]

        fig, axes = posterior.make_triangle_plot(
            keys,
            color="red",
            label="Test",
            N=50,
            fill=True,
            plot_density=True,
            truth=[30.0, 25.0],
            percentiles=[5, 95],
            grid=True,
        )

        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)

    def test_make_spindisk_plot(self, sample_rift_file, matplotlib_backend):
        """Test spin disk plot creation with mock data."""
        posterior = RIFTPosterior(sample_rift_file)

        fig = posterior.make_spindisk_plot(color="green", label="Spin Disk Test")

        # Should return figure
        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)


# The End
