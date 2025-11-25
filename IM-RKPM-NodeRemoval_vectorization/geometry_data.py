from dataclasses import dataclass
from typing import List, Union

from common import np


@dataclass
class GeometryData:
    """Geometry data and parameters for the simulation"""

    # Domain boundaries
    x_min: float = -10e-6
    x_max: float = 10e-6
    y_min: float = -10e-6
    y_max: float = 10e-6

    # Grain configuration
    single_grain: str = "False"  # "True" or "False" as string to match main.py
    angle: Union[float, List[float], np.ndarray] = None
    n_boundaries: int = 4

    # Gauss integration points for rectangular domain
    x_G_domain_rec: List[List[float]] = None
    weight_G_domain_rec: List[float] = None

    # Gauss integration points for triangular domain
    x_G_domain_tri: List[List[float]] = None
    weight_G_domain_tri: List[float] = None

    # Gauss integration points for boundaries
    x_G_boundary: List[float] = None
    weight_G_boundary: List[float] = None

    def __post_init__(self):
        """Initialize default values for Gauss points if not provided"""

        if self.x_G_domain_rec is None:
            # 2D Gauss points in neutral coordinate system for rectangular domain
            self.x_G_domain_rec = [
                [-(3**0.5) / 3, -(3**0.5) / 3],
                [-(3**0.5) / 3, 3**0.5 / 3],
                [3**0.5 / 3, -(3**0.5) / 3],
                [3**0.5 / 3, 3**0.5 / 3],
            ]

        if self.weight_G_domain_rec is None:
            # Weights for 2D Gauss points (rectangular)
            self.weight_G_domain_rec = [1.0, 1.0, 1.0, 1.0]

        if self.x_G_domain_tri is None:
            # 2D Gauss points for triangular domain
            self.x_G_domain_tri = [
                [1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
            ]

        if self.weight_G_domain_tri is None:
            # Weights for triangular Gauss points
            self.weight_G_domain_tri = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        if self.x_G_boundary is None:
            # 1D Gauss points for boundaries
            self.x_G_boundary = [
                -((3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
                -((3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
                (3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
                (3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
            ]

        if self.weight_G_boundary is None:
            # Weights for 1D boundary Gauss points
            self.weight_G_boundary = [
                0.5 - 30**0.5 / 36,
                0.5 + 30**0.5 / 36,
                0.5 + 30**0.5 / 36,
                0.5 - 30**0.5 / 36,
            ]


# Predefined geometry configurations

# Single grain configuration
single_grain_geometry = GeometryData(single_grain="True", angle=0.0, n_boundaries=4)

# Multi-grain configuration
multi_grain_angle = [
    26.0,
    np.pi,
    75.0,
    np.pi / 4.0,
    121.0,
    np.pi * 2.0 / 3.0,
    149.0,
    np.pi / 2.0,
    90.0,
    np.pi / 3.0,
    81.0,
    np.pi / 4.0,
    37.0,
    np.pi * 2.0 / 3.0,
    110.0,
    0.0,
]

multi_grain_geometry = GeometryData(
    single_grain="False", angle=multi_grain_angle, n_boundaries=4
)

# Default geometry (multi-grain)
default_geometry = multi_grain_geometry
