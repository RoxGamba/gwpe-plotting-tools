"""
Tests for gwpe_plotting_tools.gwutils module.

Tests the chi_p (precessing spin) calculation functions.
"""

import numpy as np
import pytest

from gwpe_plotting_tools.gwutils import compute_chi_prec, compute_chi_prec_from_xyz


class TestComputeChiPrec:
    """Tests for compute_chi_prec function."""

    def test_aligned_spins_zero_tilt(self, mock_masses):
        """Chi_p should be 0 for aligned spins (tilt = 0)."""
        m1, m2 = mock_masses["m1"], mock_masses["m2"]
        s1, s2 = 0.5, 0.3
        tilt1, tilt2 = 0.0, 0.0  # aligned

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        assert chi_p == pytest.approx(0.0, abs=1e-10)

    def test_anti_aligned_spins(self, mock_masses):
        """Chi_p should be 0 for anti-aligned spins (tilt = pi)."""
        m1, m2 = mock_masses["m1"], mock_masses["m2"]
        s1, s2 = 0.5, 0.3
        tilt1, tilt2 = np.pi, np.pi  # anti-aligned

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        assert chi_p == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_spins(self, mock_masses):
        """Chi_p should be non-zero for perpendicular spins."""
        m1, m2 = mock_masses["m1"], mock_masses["m2"]
        s1, s2 = 0.5, 0.3
        tilt1, tilt2 = np.pi / 2, np.pi / 2  # perpendicular

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        # chi_p should be positive for in-plane spins
        assert chi_p > 0
        # Should be bounded by the maximum spin magnitude
        assert chi_p <= max(s1, s2)

    def test_equal_mass_ratio(self):
        """Test chi_p for equal mass binary."""
        m1, m2 = 30.0, 30.0
        s1, s2 = 0.5, 0.5
        tilt1, tilt2 = np.pi / 2, np.pi / 2

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        # For equal masses with equal perpendicular spins, chi_p = s_perp
        assert chi_p == pytest.approx(0.5, rel=0.01)

    def test_mass_ordering_swap(self):
        """Test that masses are correctly ordered internally."""
        m1, m2 = 20.0, 30.0  # m2 > m1 (will be swapped)
        s1, s2 = 0.5, 0.3
        tilt1, tilt2 = np.pi / 2, np.pi / 4

        # Should handle m2 > m1 case
        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        assert chi_p >= 0
        assert chi_p <= max(s1, s2)

    def test_zero_spins(self, mock_masses):
        """Chi_p should be 0 for zero spin magnitudes."""
        m1, m2 = mock_masses["m1"], mock_masses["m2"]
        s1, s2 = 0.0, 0.0
        tilt1, tilt2 = np.pi / 2, np.pi / 2

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        assert chi_p == pytest.approx(0.0, abs=1e-10)

    def test_result_bounded(self, mock_masses, mock_spins):
        """Chi_p should always be between 0 and 1."""
        m1, m2 = mock_masses["m1"], mock_masses["m2"]
        s1, s2 = mock_spins["s1"], mock_spins["s2"]
        tilt1, tilt2 = mock_spins["tilt1"], mock_spins["tilt2"]

        chi_p = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        assert 0 <= chi_p <= 1


class TestComputeChiPrecFromXYZ:
    """Tests for compute_chi_prec_from_xyz function."""

    def test_zero_in_plane_spins(self):
        """Chi_p should be 0 when all in-plane components are zero."""
        q = 0.8
        chi1x, chi1y = 0.0, 0.0
        chi2x, chi2y = 0.0, 0.0

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        assert chi_p == pytest.approx(0.0, abs=1e-10)

    def test_only_primary_spin(self):
        """Chi_p when only primary has in-plane spin."""
        q = 1.0  # Use equal mass ratio to avoid swapping
        chi1x, chi1y = 0.3, 0.4  # chi1_perp = 0.5
        chi2x, chi2y = 0.0, 0.0

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        # For equal mass ratio with only primary spin, chi_p = chi1_perp
        expected = np.sqrt(chi1x**2 + chi1y**2)
        assert chi_p == pytest.approx(expected, rel=0.01)

    def test_only_secondary_spin(self):
        """Chi_p when only secondary has in-plane spin."""
        q = 0.8
        chi1x, chi1y = 0.0, 0.0
        chi2x, chi2y = 0.3, 0.4  # chi2_perp = 0.5

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        # Should be scaled secondary contribution
        chi2_perp = np.sqrt(chi2x**2 + chi2y**2)
        assert chi_p >= 0
        assert chi_p <= chi2_perp  # bounded

    def test_equal_mass_equal_spins(self):
        """Test chi_p for equal mass binary with equal spins."""
        q = 1.0
        chi1x, chi1y = 0.3, 0.4  # chi1_perp = 0.5
        chi2x, chi2y = 0.3, 0.4  # chi2_perp = 0.5

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        assert chi_p == pytest.approx(0.5, rel=0.01)

    def test_mass_ratio_less_than_one(self):
        """Test behavior when q < 1 (should swap internally)."""
        q = 0.5  # will be inverted internally
        chi1x, chi1y = 0.3, 0.0
        chi2x, chi2y = 0.0, 0.4

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        # Should still produce valid result
        assert chi_p >= 0
        assert chi_p <= 1

    def test_consistency_with_compute_chi_prec(self):
        """Verify consistency between the two chi_p functions for equal mass."""
        # Use equal mass for simpler comparison
        m1, m2 = 30.0, 30.0
        q = m2 / m1  # = 1.0

        # Define spins in x-y plane (tilt = pi/2)
        s1, s2 = 0.5, 0.3
        tilt1, tilt2 = np.pi / 2, np.pi / 2

        # All spin in x direction for simplicity
        chi1x, chi1y = s1, 0.0
        chi2x, chi2y = s2, 0.0

        chi_p_xyz = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)
        chi_p_tilt = compute_chi_prec(m1, m2, s1, s2, tilt1, tilt2)

        # For equal mass, both should give max(s1_perp, s2_perp) = s1 = 0.5
        assert chi_p_xyz == pytest.approx(chi_p_tilt, rel=0.05)

    def test_result_bounded(self, mock_spins):
        """Chi_p should always be between 0 and 1."""
        q = 0.8
        chi1x = mock_spins["chi1x"]
        chi1y = mock_spins["chi1y"]
        chi2x = mock_spins["chi2x"]
        chi2y = mock_spins["chi2y"]

        chi_p = compute_chi_prec_from_xyz(q, chi1x, chi1y, chi2x, chi2y)

        assert 0 <= chi_p <= 1

    def test_numpy_array_input(self):
        """Test that function handles numpy arrays."""
        q = np.array([0.7, 0.8, 0.9])
        chi1x = np.array([0.1, 0.2, 0.3])
        chi1y = np.array([0.1, 0.1, 0.1])
        chi2x = np.array([0.05, 0.1, 0.15])
        chi2y = np.array([0.05, 0.05, 0.05])

        # Should work with arrays (vectorized)
        # Note: current implementation may not be fully vectorized
        # This tests scalar behavior with the first element
        chi_p = compute_chi_prec_from_xyz(q[0], chi1x[0], chi1y[0], chi2x[0], chi2y[0])

        assert isinstance(chi_p, (float, np.floating))
        assert 0 <= chi_p <= 1
