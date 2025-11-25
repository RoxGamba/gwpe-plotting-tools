"""
Tests for gwpe_plotting_tools.logging_config module.
"""

import logging

import pytest

from gwpe_plotting_tools.logging_config import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function.
    
    Note: logging.basicConfig only works on the first call unless
    the root logger has no handlers. These tests verify the function
    interface rather than trying to reset logging state between tests.
    """

    def test_setup_logging_can_be_called(self):
        """Test that setup_logging can be called without errors."""
        # This should not raise any exceptions
        setup_logging()

    def test_setup_logging_with_debug_level(self):
        """Test that setup_logging accepts DEBUG level."""
        # This should not raise any exceptions
        setup_logging(level="DEBUG")

    def test_setup_logging_with_warning_level(self):
        """Test that setup_logging accepts WARNING level."""
        setup_logging(level="WARNING")

    def test_setup_logging_with_error_level(self):
        """Test that setup_logging accepts ERROR level."""
        setup_logging(level="ERROR")

    def test_setup_logging_with_critical_level(self):
        """Test that setup_logging accepts CRITICAL level."""
        setup_logging(level="CRITICAL")

    def test_setup_logging_case_insensitive(self):
        """Test that level parameter is case-insensitive."""
        # These should not raise any exceptions
        setup_logging(level="debug")
        setup_logging(level="Debug")
        setup_logging(level="DEBUG")

    def test_setup_logging_custom_format(self):
        """Test that custom format string is accepted."""
        custom_format = "%(levelname)s - %(message)s"
        setup_logging(format_string=custom_format)

    def test_setup_logging_custom_datefmt(self):
        """Test that custom date format is accepted."""
        custom_datefmt = "%H:%M:%S"
        setup_logging(datefmt=custom_datefmt)

    def test_setup_logging_all_parameters(self):
        """Test setting all parameters at once."""
        setup_logging(
            level="DEBUG",
            format_string="%(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d",
        )

    def test_setup_logging_invalid_level_raises(self):
        """Test that invalid level raises AttributeError."""
        with pytest.raises(AttributeError):
            setup_logging(level="INVALID_LEVEL")

    def test_setup_logging_returns_none(self):
        """Test that setup_logging returns None."""
        result = setup_logging()
        assert result is None
