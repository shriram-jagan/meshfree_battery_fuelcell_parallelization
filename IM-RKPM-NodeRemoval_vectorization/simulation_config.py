"""
Simulation configuration module containing all simulation parameters
and settings that are not material properties or geometry data.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TimeConfig:
    """Time stepping configuration"""

    t: float = 1.0  # Total simulation time (seconds)
    nt: int = 2  # Number of time steps

    @property
    def dt(self) -> float:
        """Time step size"""
        return self.t / self.nt


@dataclass
class NumericalConfig:
    """Numerical method configuration"""

    differential_method: str = "direct"  # 'implicit' or 'direct'
    integral_method: str = "gauss"  # Integration method
    c: int = 2  # Support size for RKPM


@dataclass
class ModelConfig:
    """Model configuration switches"""

    IM_RKPM: str = "True"  # If it is interfacial modified RKPM
    Node_removal: str = "True"  # Node removal feature
    damage_model: str = "ON"  # "ON" or "OFF"


@dataclass
class InitialConditions:
    """Initial conditions for the simulation"""

    ini_charge_state: float = 0.92  # Initial charge state
    ini_potential: float = 3.712  # Initial potential (V)


@dataclass
class ConvergenceConfig:
    """Convergence criteria configuration"""

    dc_threshold: float = 1.0e-9  # Concentration convergence threshold
    dphi_threshold: float = 1.0e-9  # Potential convergence threshold
    max_newton_iter: int = 10  # Maximum Newton iterations


@dataclass
class BasisVectors:
    """Basis vectors for shape function computation"""

    HT0: np.ndarray = field(
        default_factory=lambda: np.array([1, 0, 0], dtype=np.float64)
    )
    HT1: np.ndarray = field(
        default_factory=lambda: np.array([0, -1, 0], dtype=np.float64)
    )
    HT2: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, -1], dtype=np.float64)
    )


@dataclass
class SimulationConfig:
    """Main simulation configuration class combining all config sections"""

    time: TimeConfig = field(default_factory=TimeConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    basis_vectors: BasisVectors = field(default_factory=BasisVectors)

    def validate(self):
        """Validate configuration consistency"""
        # If IM_RKPM is True, differential method must be 'direct'
        if (
            self.model.IM_RKPM == "True"
            and self.numerical.differential_method != "direct"
        ):
            raise ValueError(
                "When IM_RKPM is True, differential_method must be 'direct'"
            )

        # Check valid options
        if self.numerical.differential_method not in ["direct", "implicit"]:
            raise ValueError(
                f"Invalid differential_method: {self.numerical.differential_method}"
            )

        if self.model.damage_model not in ["ON", "OFF"]:
            raise ValueError(f"Invalid damage_model: {self.model.damage_model}")

        if self.model.IM_RKPM not in ["True", "False"]:
            raise ValueError(f"Invalid IM_RKPM: {self.model.IM_RKPM}")

        if self.model.Node_removal not in ["True", "False"]:
            raise ValueError(f"Invalid Node_removal: {self.model.Node_removal}")

        return True


# Predefined configurations

# Default configuration
default_config = SimulationConfig()

# Fast test configuration (fewer time steps)
fast_test_config = SimulationConfig(
    time=TimeConfig(t=0.1, nt=1), convergence=ConvergenceConfig(max_newton_iter=5)
)

# No damage configuration
no_damage_config = SimulationConfig(
    model=ModelConfig(damage_model="OFF", Node_removal="False")
)

# Standard RKPM configuration (not interfacial modified)
standard_rkpm_config = SimulationConfig(
    model=ModelConfig(IM_RKPM="False", Node_removal="False"),
    numerical=NumericalConfig(differential_method="implicit"),
)

# High accuracy configuration
high_accuracy_config = SimulationConfig(
    time=TimeConfig(t=10.0, nt=100),
    convergence=ConvergenceConfig(
        dc_threshold=1.0e-12, dphi_threshold=1.0e-12, max_newton_iter=20
    ),
)
