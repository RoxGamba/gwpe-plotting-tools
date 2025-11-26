"""
Tests for compatibility with external libraries.

These tests verify that gwpe-plotting-tools works correctly with
the latest versions of bilby, corner, and pesummary.
"""

import pytest


class TestCornerCompatibility:
    """Tests for corner library compatibility."""

    def test_corner_import(self):
        """Test that corner can be imported."""
        import corner

        assert corner is not None

    def test_corner_version(self):
        """Test corner version is recent enough."""
        import corner

        version = corner.__version__
        major = int(version.split(".")[0])

        # Should be version 2.x or higher
        assert major >= 2

    def test_corner_basic_plot(self, matplotlib_backend):
        """Test that corner can create a basic plot."""
        import numpy as np
        import corner

        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, 2)

        # Create corner plot
        fig = corner.corner(data)

        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)


class TestPesummaryCompatibility:
    """Tests for pesummary library compatibility."""

    def test_pesummary_import(self):
        """Test that pesummary can be imported."""
        import pesummary

        assert pesummary is not None

    def test_pesummary_publication_plots_import(self):
        """Test that pesummary publication plots can be imported."""
        from pesummary.core.plots.publication import _triangle_plot, _triangle_axes

        assert _triangle_plot is not None
        assert _triangle_axes is not None

    def test_pesummary_triangle_axes(self, matplotlib_backend):
        """Test that pesummary triangle axes work."""
        from pesummary.core.plots.publication import _triangle_axes

        fig, ax1, temp, ax2, ax3 = _triangle_axes(figsize=(5, 5))

        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        assert ax3 is not None

        # Clean up
        matplotlib_backend.close(fig)

    def test_pesummary_triangle_plot(self, matplotlib_backend):
        """Test that pesummary triangle plot works."""
        import numpy as np
        from pesummary.core.plots.publication import _triangle_plot, _triangle_axes

        # Create sample data
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Create axes
        fig, ax1, temp, ax2, ax3 = _triangle_axes(figsize=(5, 5))
        temp.remove()

        # Create triangle plot
        fig, ax1, ax2, ax3 = _triangle_plot(
            x=x,
            y=y,
            xlabel="X",
            ylabel="Y",
            colors=["blue"],
            labels=["Test"],
            fig=fig,
            axes=(ax1, ax2, ax3),
        )

        assert fig is not None

        # Clean up
        matplotlib_backend.close(fig)


class TestBilbyCompatibility:
    """Tests for bilby library compatibility."""

    def test_bilby_import(self):
        """Test that bilby can be imported."""
        import bilby

        assert bilby is not None

    def test_bilby_version(self):
        """Test bilby version is recent enough."""
        import bilby

        version = bilby.__version__
        major = int(version.split(".")[0])

        # Should be version 2.x or higher
        assert major >= 2

    def test_bilby_gw_modules(self):
        """Test that bilby GW modules can be imported."""
        from bilby.gw.prior import BBHPriorDict
        from bilby.gw.result import CBCResult

        assert BBHPriorDict is not None
        assert CBCResult is not None

    def test_bilby_prior_dict_creation(self):
        """Test that BBHPriorDict can be created."""
        from bilby.gw.prior import BBHPriorDict

        prior = BBHPriorDict()

        assert prior is not None


class TestScipyCompatibility:
    """Tests for scipy library compatibility."""

    def test_scipy_import(self):
        """Test that scipy can be imported."""
        import scipy

        assert scipy is not None

    def test_scipy_stats_kde(self):
        """Test that scipy gaussian_kde works."""
        import numpy as np
        from scipy.stats import gaussian_kde

        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100)

        # Create KDE
        kde = gaussian_kde(data)

        # Evaluate KDE at some points
        result = kde([0.0, 0.5, 1.0])

        assert result is not None
        assert len(result) == 3


class TestNumpyCompatibility:
    """Tests for numpy library compatibility."""

    def test_numpy_import(self):
        """Test that numpy can be imported."""
        import numpy as np

        assert np is not None

    def test_numpy_version(self):
        """Test numpy version is recent enough."""
        import numpy as np

        version = np.__version__
        major = int(version.split(".")[0])

        # Should be version 1.x or 2.x
        assert major >= 1


class TestMatplotlibCompatibility:
    """Tests for matplotlib library compatibility."""

    def test_matplotlib_import(self):
        """Test that matplotlib can be imported."""
        import matplotlib
        import matplotlib.pyplot as plt

        assert matplotlib is not None
        assert plt is not None

    def test_matplotlib_backend_agg(self, matplotlib_backend):
        """Test that Agg backend works."""
        import matplotlib

        # Backend should be Agg for testing
        backend = matplotlib.get_backend().lower()
        assert "agg" in backend
