"""
Unit tests for the config module.
"""

import config
import numpy as np
import pytest


class TestConfig:
    """Test suite for configuration module."""

    def test_geometry_settings(self):
        """Test geometry configuration settings."""
        assert isinstance(config.SINGLE_GRAIN, bool)
        assert config.DIMENSION in [2, 3]

    def test_analysis_settings(self):
        """Test analysis configuration settings."""
        assert isinstance(config.IM_RKPM, bool)
        assert config.STUDIED_PHYSICS in ["fuel cell", "battery"]
        assert config.DAMAGE_MODEL in ["ON", "OFF"]
        assert isinstance(config.DELTA_POINT_SOURCE, bool)

    def test_output_control_flags(self):
        """Test output control flags."""
        assert isinstance(config.ENABLE_PLOTTING, bool)
        assert isinstance(config.ENABLE_FILE_IO, bool)

    def test_numerical_methods(self):
        """Test numerical method settings."""
        assert config.DIFFERENTIAL_METHOD in ["implicit", "direct"]
        assert config.INTEGRAL_METHOD == "gauss"

    def test_domain_dimensions(self):
        """Test domain dimension values."""
        # Check that domain boundaries are defined
        assert hasattr(config, "X_MIN")
        assert hasattr(config, "X_MAX")
        assert hasattr(config, "Y_MIN")
        assert hasattr(config, "Y_MAX")
        assert hasattr(config, "Z_MIN")
        assert hasattr(config, "Z_MAX")

        # Check that max > min for all dimensions
        assert config.X_MAX > config.X_MIN
        assert config.Y_MAX > config.Y_MIN
        assert config.Z_MAX > config.Z_MIN

        # Check that values are reasonable (in meters, microscale)
        assert config.X_MAX - config.X_MIN < 1e-3  # Less than 1mm
        assert config.Y_MAX - config.Y_MIN < 1e-3
        assert config.Z_MAX - config.Z_MIN < 1e-3

    def test_physical_constants(self):
        """Test physical constants."""
        # Faraday constant
        assert config.FARADAY_CONSTANT == pytest.approx(9.6485e4, rel=1e-4)

        # Gas constant
        assert config.GAS_CONSTANT == pytest.approx(8.3145, rel=1e-4)

    def test_diffusion_coefficients(self):
        """Test diffusion coefficient values."""
        # Check that diffusion coefficients are positive
        assert config.DIFFUSION_ELECTROLYTE > 0
        assert config.DIFFUSION_ELECTRODE > 0
        assert config.DIFFUSION_PORE > 0

        # Check relative magnitudes (typical for fuel cells)
        assert config.DIFFUSION_ELECTROLYTE > config.DIFFUSION_ELECTRODE
        assert config.DIFFUSION_PORE > config.DIFFUSION_ELECTRODE

    def test_electrochemical_parameters(self):
        """Test electrochemical parameter values."""
        # Exchange current densities should be positive
        assert config.I_0 > 0
        assert config.I_0_SOLID > 0

        # Temperature should be reasonable (in Kelvin)
        assert config.TEMPERATURE > 273  # Above 0°C
        assert config.TEMPERATURE < 2000  # Below 1727°C

        # Potentials
        assert config.E_0 > 0
        assert config.V_APP > 0

        # Boundary concentrations should be positive
        assert config.C_BOUNDARY > 0
        assert config.C_BOUNDARY_PORE > 0

    def test_mechanical_properties_electrolyte(self):
        """Test mechanical properties for electrolyte."""
        # Young's modulus should be positive and in reasonable range
        assert config.E_ELECTROLYTE > 0
        assert config.E_ELECTROLYTE > 1e9  # Greater than 1 GPa
        assert config.E_ELECTROLYTE < 1e12  # Less than 1 TPa

        # Poisson's ratio should be between -1 and 0.5
        assert -1 < config.NU_ELECTROLYTE < 0.5

        # Lamé constants
        assert config.LAMBDA_MECHANICAL_ELECTROLYTE > 0
        assert config.MU_ELECTROLYTE > 0

        # Verify Lamé constant calculations
        nu = config.NU_ELECTROLYTE
        E = config.E_ELECTROLYTE
        expected_lambda = E * nu / (1 + nu) / (1 - 2 * nu)
        expected_mu = E / 2 / (1 + nu)

        assert config.LAMBDA_MECHANICAL_ELECTROLYTE == pytest.approx(expected_lambda)
        assert config.MU_ELECTROLYTE == pytest.approx(expected_mu)

    def test_mechanical_properties_electrode(self):
        """Test mechanical properties for electrode."""
        # Young's modulus should be positive and in reasonable range
        assert config.E_ELECTRODE > 0
        assert config.E_ELECTRODE > 1e9  # Greater than 1 GPa
        assert config.E_ELECTRODE < 1e12  # Less than 1 TPa

        # Poisson's ratio should be between -1 and 0.5
        assert -1 < config.NU_ELECTRODE < 0.5

        # Lamé constants
        assert config.LAMBDA_MECHANICAL_ELECTRODE > 0
        assert config.MU_ELECTRODE > 0

        # Verify Lamé constant calculations
        nu = config.NU_ELECTRODE
        E = config.E_ELECTRODE
        expected_lambda = E * nu / (1 + nu) / (1 - 2 * nu)
        expected_mu = E / 2 / (1 + nu)

        assert config.LAMBDA_MECHANICAL_ELECTRODE == pytest.approx(expected_lambda)
        assert config.MU_ELECTRODE == pytest.approx(expected_mu)

    def test_expansion_coefficient(self):
        """Test thermal expansion coefficient."""
        # Should be positive and in reasonable range
        assert config.BETA_FUELCELL_EXPANSION_COEFFICIENT > 0
        assert (
            config.BETA_FUELCELL_EXPANSION_COEFFICIENT < 1e-3
        )  # Reasonable for materials

    def test_damage_parameters(self):
        """Test damage model parameters."""
        # Damage parameters should be positive
        assert config.K_I > 0
        assert config.K_F > 0

        # Final damage should be greater than initial
        assert config.K_F > config.K_I

    def test_image_file_name(self):
        """Test image file name configuration."""
        assert isinstance(config.IMAGE_FILE_NAME, str)
        assert config.IMAGE_FILE_NAME.endswith(".tif")

    def test_gauss_points_3d_cube(self):
        """Test 3D cube Gauss points and weights."""
        # Check shape
        assert config.X_G_CUBE.shape == (8, 3)
        assert config.WEIGHT_G_CUBE.shape == (8,)

        # Check that weights sum to 8 for 3D cube
        assert np.sum(config.WEIGHT_G_CUBE) == pytest.approx(8.0)

        # Check that Gauss points are within [-1, 1]
        assert np.all(np.abs(config.X_G_CUBE) <= 1.0)

    def test_gauss_points_2d_rectangle(self):
        """Test 2D rectangle Gauss points and weights."""
        # Check shape
        assert config.X_G_REC.shape == (4, 2)
        assert config.WEIGHT_G_REC.shape == (4,)

        # Check that weights sum to 4 for 2D rectangle
        assert np.sum(config.WEIGHT_G_REC) == pytest.approx(4.0)

        # Check that Gauss points are within [-1, 1]
        assert np.all(np.abs(config.X_G_REC) <= 1.0)

    def test_gauss_points_2d_triangle(self):
        """Test 2D triangle Gauss points and weights."""
        # Check shape
        assert config.X_G_TRI.shape == (3, 2)
        assert config.WEIGHT_G_TRI.shape == (3,)

        # Check that weights sum to 1 for triangle
        assert np.sum(config.WEIGHT_G_TRI) == pytest.approx(1.0)

        # Check that Gauss points are within triangle bounds
        assert np.all(config.X_G_TRI >= 0)
        assert np.all(config.X_G_TRI <= 1)

    def test_gauss_points_1d_line(self):
        """Test 1D line Gauss points and weights."""
        # Check shape
        assert config.X_G_LINE.shape == (4,)
        assert config.WEIGHT_G_LINE.shape == (4,)

        # Check that weights sum to 2 for 1D line [-1, 1]
        assert np.sum(config.WEIGHT_G_LINE) == pytest.approx(2.0)

        # Check that Gauss points are within [-1, 1]
        assert np.all(np.abs(config.X_G_LINE) <= 1.0)

    def test_gas_permeability(self):
        """Test gas permeability parameter."""
        assert config.K_GAS > 0
        assert config.K_GAS < 1  # Typically small value
