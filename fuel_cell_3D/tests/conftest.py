"""
Shared fixtures and configuration for pytest tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

# Add the parent directory to the Python path to import fuel_cell modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_3d_image(temp_dir):
    """Create a sample 3D TIFF image for testing."""
    # Create a simple 3D image with 3 phases
    # 0: pore, 1: electrolyte, 2: electrode
    img_data = np.zeros((5, 5, 5), dtype=np.uint8)

    # Create a simple pattern
    img_data[0:2, :, :] = 0  # pore phase
    img_data[2:3, :, :] = 1  # electrolyte phase
    img_data[3:5, :, :] = 2  # electrode phase

    # Save the image
    img_path = temp_dir / "test_image.tif"
    tifffile.imwrite(str(img_path), img_data)

    return str(img_path), img_data


@pytest.fixture
def sample_2d_image(temp_dir):
    """Create a sample 2D TIFF image for testing."""
    # Create a simple 2D image with 3 phases
    img_data = np.zeros((10, 10), dtype=np.uint8)

    # Create a simple pattern
    img_data[0:3, :] = 0  # pore phase
    img_data[3:7, :] = 1  # electrolyte phase
    img_data[7:10, :] = 2  # electrode phase

    # Save the image
    img_path = temp_dir / "test_2d_image.tif"
    tifffile.imwrite(str(img_path), img_data)

    return str(img_path), img_data


@pytest.fixture
def sample_config():
    """Create a sample configuration object for testing."""

    class MockConfig:
        # Geometry settings
        SINGLE_GRAIN = False
        DIMENSION = 3

        # Analysis settings
        IM_RKPM = False
        STUDIED_PHYSICS = "fuel cell"
        DAMAGE_MODEL = "OFF"
        DELTA_POINT_SOURCE = True

        # Output settings
        ENABLE_PLOTTING = False
        ENABLE_FILE_IO = False

        # Numerical methods
        DIFFERENTIAL_METHOD = "direct"
        INTEGRAL_METHOD = "gauss"

        # Domain dimensions
        X_MIN = 0.0
        X_MAX = 10e-6
        Y_MIN = 0.0
        Y_MAX = 20e-6
        Z_MIN = 0.0
        Z_MAX = 10e-6

        # Material orientation
        ANGLE = 0.0

        # Physical constants
        FARADAY_CONSTANT = 9.6485e4
        GAS_CONSTANT = 8.3145

        # Diffusion coefficients
        DIFFUSION_ELECTROLYTE = 0.035
        DIFFUSION_ELECTRODE = 1.0e-9
        DIFFUSION_PORE = 1.0e-7

        # Gas permeability
        K_GAS = 1.0e-4

        # Electrochemical parameters
        I_0 = 1.0e-1
        I_0_SOLID = 1.0e1
        TEMPERATURE = 1273.2
        E_0 = 1.0
        V_APP = 1.5
        C_BOUNDARY = 1000.0
        C_BOUNDARY_PORE = 9.572

        # Mechanical properties - Electrolyte
        E_ELECTROLYTE = 132.69e9
        NU_ELECTROLYTE = 0.33
        LAMBDA_MECHANICAL_ELECTROLYTE = (
            E_ELECTROLYTE
            * NU_ELECTROLYTE
            / (1 + NU_ELECTROLYTE)
            / (1 - 2 * NU_ELECTROLYTE)
        )
        MU_ELECTROLYTE = E_ELECTROLYTE / 2 / (1 + NU_ELECTROLYTE)

        # Mechanical properties - Electrode
        E_ELECTRODE = 130.0e9
        NU_ELECTRODE = 0.33
        LAMBDA_MECHANICAL_ELECTRODE = (
            E_ELECTRODE * NU_ELECTRODE / (1 + NU_ELECTRODE) / (1 - 2 * NU_ELECTRODE)
        )
        MU_ELECTRODE = E_ELECTRODE / 2 / (1 + NU_ELECTRODE)

        # Expansion coefficient
        BETA_FUELCELL_EXPANSION_COEFFICIENT = 4.0e-6

        # Damage parameters
        K_I = 0.0125
        K_F = 0.015

        # Image file
        IMAGE_FILE_NAME = "test_image.tif"

        # Gauss points and weights for 3D cube
        X_G_CUBE = np.array(
            [
                [-(3**0.5) / 3, -(3**0.5) / 3, -(3**0.5) / 3],
                [3**0.5 / 3, -(3**0.5) / 3, -(3**0.5) / 3],
                [-(3**0.5) / 3, 3**0.5 / 3, -(3**0.5) / 3],
                [3**0.5 / 3, 3**0.5 / 3, -(3**0.5) / 3],
                [-(3**0.5) / 3, -(3**0.5) / 3, 3**0.5 / 3],
                [3**0.5 / 3, -(3**0.5) / 3, 3**0.5 / 3],
                [-(3**0.5) / 3, 3**0.5 / 3, 3**0.5 / 3],
                [3**0.5 / 3, 3**0.5 / 3, 3**0.5 / 3],
            ]
        )

        WEIGHT_G_CUBE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Gauss points for 2D rectangle
        X_G_REC = np.array(
            [
                [-(3**0.5) / 3, -(3**0.5) / 3],
                [-(3**0.5) / 3, 3**0.5 / 3],
                [3**0.5 / 3, -(3**0.5) / 3],
                [3**0.5 / 3, 3**0.5 / 3],
            ]
        )

        WEIGHT_G_REC = np.array([1.0, 1.0, 1.0, 1.0])

        # Add triangle Gauss points that were missing
        X_G_TRI = np.array(
            [
                [1.0 / 6.0, 2.0 / 3.0],
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
            ]
        )
        WEIGHT_G_TRI = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # Add 1D line Gauss points
        X_G_LINE = np.array(
            [
                -((3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
                -((3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
                (3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
                (3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
            ]
        )
        WEIGHT_G_LINE = np.array(
            [
                0.5 - 30**0.5 / 36,
                0.5 + 30**0.5 / 36,
                0.5 + 30**0.5 / 36,
                0.5 - 30**0.5 / 36,
            ]
        )

    return MockConfig()


@pytest.fixture
def sample_concentration():
    """Create sample concentration data for testing."""
    # Create concentration values between 0 and 1
    x_values = np.linspace(0.1, 0.9, 10)
    return x_values


@pytest.fixture
def sample_damage():
    """Create sample damage data for testing."""
    # Create damage values between 0 and 0.9
    damage_values = np.linspace(0.0, 0.8, 10)
    return damage_values
