"""Tests for BaseAnalyzer."""

import pytest

from py_oats_overhaul.reader.trajectory import TrajectoryData
from py_oats_overhaul.analyzers.base import BaseAnalyzer


def test_base_analyzer_instantiation(traj_data_minimal):
    """BaseAnalyzer stores trajectory and default name."""
    a = BaseAnalyzer(traj_data_minimal)
    assert a.trajectory is traj_data_minimal
    assert a.name == "base_analyzer"


def test_base_analyzer_custom_name(traj_data_minimal):
    """BaseAnalyzer accepts custom name."""
    a = BaseAnalyzer(traj_data_minimal, name="custom")
    assert a.name == "custom"


def test_base_analyzer_analyze_raises(traj_data_minimal):
    """BaseAnalyzer.analyze() raises NotImplementedError."""
    a = BaseAnalyzer(traj_data_minimal)
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        a.analyze()
