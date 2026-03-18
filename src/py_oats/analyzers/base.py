""" Base class for all analyzers. """
from py_oats.io.trajectory import TrajectoryData

class BaseAnalyzer:
    """
    Base class for all analyzers.
    """
    def __init__(self, trajectory: TrajectoryData, name: str = "base_analyzer") -> None:
        """
        Args:
            trajectory: TrajectoryData, the trajectory to analyze
            name: str, the name of the analyzer (default: "base_analyzer")
        """
        self.trajectory = trajectory
        self.name = name

    def analyze(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")