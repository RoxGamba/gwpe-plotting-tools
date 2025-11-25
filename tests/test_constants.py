"""
Tests for gwpe_plotting_tools.constants module.

Tests parameter name mappings and LaTeX labels.
"""

import pytest

from gwpe_plotting_tools.constants import RIFT_TO_BILBY, MDC_TO_BILBY, KEYS_LATEX


class TestRiftToBilbyMapping:
    """Tests for RIFT_TO_BILBY parameter mapping."""

    def test_mass_parameters_mapped(self):
        """Test that mass parameters are correctly mapped."""
        assert RIFT_TO_BILBY["m1"] == "mass_1"
        assert RIFT_TO_BILBY["m2"] == "mass_2"
        assert RIFT_TO_BILBY["mc"] == "chirp_mass"
        assert RIFT_TO_BILBY["mtotal"] == "total_mass"
        assert RIFT_TO_BILBY["eta"] == "symmetric_mass_ratio"
        assert RIFT_TO_BILBY["q"] == "mass_ratio"

    def test_spin_parameters_mapped(self):
        """Test that spin parameters are correctly mapped."""
        assert RIFT_TO_BILBY["a1x"] == "spin_1x"
        assert RIFT_TO_BILBY["a1y"] == "spin_1y"
        assert RIFT_TO_BILBY["a1z"] == "spin_1z"
        assert RIFT_TO_BILBY["a2x"] == "spin_2x"
        assert RIFT_TO_BILBY["a2y"] == "spin_2y"
        assert RIFT_TO_BILBY["a2z"] == "spin_2z"
        assert RIFT_TO_BILBY["chi_eff"] == "chi_eff"
        assert RIFT_TO_BILBY["chi_p"] == "chi_p"

    def test_sky_localization_mapped(self):
        """Test that sky localization parameters are correctly mapped."""
        assert RIFT_TO_BILBY["ra"] == "ra"
        assert RIFT_TO_BILBY["dec"] == "dec"
        assert RIFT_TO_BILBY["distance"] == "luminosity_distance"

    def test_orientation_parameters_mapped(self):
        """Test that orientation parameters are correctly mapped."""
        assert RIFT_TO_BILBY["incl"] == "iota"
        assert RIFT_TO_BILBY["psi"] == "psi"
        assert RIFT_TO_BILBY["phiorb"] == "phase"

    def test_time_parameters_mapped(self):
        """Test that time parameters are correctly mapped."""
        assert RIFT_TO_BILBY["time"] == "geocent_time"

    def test_likelihood_mapped(self):
        """Test that likelihood parameters are correctly mapped."""
        assert RIFT_TO_BILBY["lnL"] == "log_likelihood"

    def test_source_frame_masses_mapped(self):
        """Test that source frame masses are correctly mapped."""
        assert RIFT_TO_BILBY["m1_source"] == "mass_1_source"
        assert RIFT_TO_BILBY["m2_source"] == "mass_2_source"
        assert RIFT_TO_BILBY["mc_source"] == "chirp_mass_source"
        assert RIFT_TO_BILBY["mtotal_source"] == "total_mass_source"

    def test_eccentricity_mapped(self):
        """Test that eccentricity parameters are correctly mapped."""
        assert RIFT_TO_BILBY["eccentricity"] == "eccentricity"
        assert RIFT_TO_BILBY["meanPerAno"] == "mean_per_ano"


class TestMdcToBilbyMapping:
    """Tests for MDC_TO_BILBY parameter mapping."""

    def test_inherits_rift_mapping(self):
        """MDC mapping should include all RIFT mappings."""
        for key, value in RIFT_TO_BILBY.items():
            assert MDC_TO_BILBY[key] == value

    def test_additional_mdc_parameters(self):
        """Test MDC-specific parameter mappings."""
        assert MDC_TO_BILBY["tref"] == "geocent_time"
        assert MDC_TO_BILBY["dist"] == "luminosity_distance"
        assert MDC_TO_BILBY["total_mass"] == "total_mass"
        assert MDC_TO_BILBY["mass_ratio"] == "mass_ratio"

    def test_gr_deviation_parameters(self):
        """Test GR deviation parameters are mapped."""
        assert MDC_TO_BILBY["delta_taulm0"] == "delta_taulm0"
        assert MDC_TO_BILBY["delta_omglm0"] == "delta_omglm0"
        assert MDC_TO_BILBY["delta_Mbhf"] == "delta_Mbhf"
        assert MDC_TO_BILBY["delta_abhf"] == "delta_abhf"


class TestKeysLatex:
    """Tests for KEYS_LATEX LaTeX label mapping."""

    def test_mass_labels_have_latex(self):
        """Test that mass parameters have LaTeX labels."""
        assert "mass_1" in KEYS_LATEX
        assert "mass_2" in KEYS_LATEX
        assert "chirp_mass" in KEYS_LATEX
        assert "total_mass" in KEYS_LATEX
        assert "mass_ratio" in KEYS_LATEX
        assert "symmetric_mass_ratio" in KEYS_LATEX

    def test_spin_labels_have_latex(self):
        """Test that spin parameters have LaTeX labels."""
        assert "chi_eff" in KEYS_LATEX
        assert "chi_p" in KEYS_LATEX
        assert "chi_1" in KEYS_LATEX
        assert "chi_2" in KEYS_LATEX

    def test_distance_labels_have_latex(self):
        """Test that distance parameters have LaTeX labels."""
        assert "luminosity_distance" in KEYS_LATEX

    def test_orientation_labels_have_latex(self):
        """Test that orientation parameters have LaTeX labels."""
        assert "iota" in KEYS_LATEX
        assert "psi" in KEYS_LATEX
        assert "phase" in KEYS_LATEX
        assert "ra" in KEYS_LATEX
        assert "dec" in KEYS_LATEX

    def test_latex_format_valid(self):
        """Test that LaTeX labels use valid LaTeX format."""
        for key, label in KEYS_LATEX.items():
            # Labels should contain $ for math mode
            assert "$" in label, f"Label for {key} should contain LaTeX math mode"

    def test_latex_labels_are_strings(self):
        """Test that all LaTeX labels are strings."""
        for key, label in KEYS_LATEX.items():
            assert isinstance(label, str), f"Label for {key} should be a string"

    def test_eccentricity_labels_have_latex(self):
        """Test that eccentricity parameters have LaTeX labels."""
        assert "eccentricity" in KEYS_LATEX
        assert "mean_per_ano" in KEYS_LATEX

    def test_gr_deviation_labels_have_latex(self):
        """Test that GR deviation parameters have LaTeX labels."""
        assert "delta_taulm0" in KEYS_LATEX
        assert "delta_omglm0" in KEYS_LATEX
        assert "delta_abhf" in KEYS_LATEX
        assert "delta_Mbhf" in KEYS_LATEX
