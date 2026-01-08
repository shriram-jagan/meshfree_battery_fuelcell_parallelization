"""
Integration tests for the fuel cell 3D simulation workflow.
"""

import os
import sys

import pytest

sys.path.insert(0, "..")
from common import np, sp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from define_diffusion_matrix_form import diffusion_matrix_fuel_cell
from define_mechanical_stiffness_matrix import (
    mechanical_C_tensor_3d,
    mechanical_force_matrix_3d,
    mechanical_stiffness_matrix_3d_fuel_cell,
)
from read_image import read_in_image


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_image_to_diffusion_matrix_workflow(self, sample_3d_image):
        """Test workflow from image reading to diffusion matrix construction."""
        img_path, _ = sample_3d_image

        # Step 1: Read image
        img_data, unique_ids, num_pixels = read_in_image(img_path, "fuel cell", 3)

        assert img_data is not None
        assert len(unique_ids) == 3  # Three phases
        assert len(num_pixels) == 3  # 3D

        # Step 2: Setup geometry (simplified)
        n_nodes = np.prod(num_pixels)
        n_G = 8  # Gauss points per element

        # Create mock shape functions
        phi = np.random.rand(n_G, n_nodes) * 0.1
        grad_phi = np.random.rand(n_G, n_nodes, 3) * 0.1
        Jwei = np.ones(n_G) * (1.0 / n_G)
        indx = np.ones(n_G)

        # Step 3: Create diffusion matrix for each phase
        diff_coefficients = {
            0: 1.0e-7,  # pore
            1: 0.035,  # electrolyte
            2: 1.0e-9,  # electrode
        }

        for phase_id, diff_coeff in diff_coefficients.items():
            D_matrix = diffusion_matrix_fuel_cell(
                n_nodes, n_G, Jwei, grad_phi, diff_coeff, indx
            )

            # Verify matrix properties
            assert D_matrix.shape == (n_nodes, n_nodes)
            assert sp.issparse(D_matrix)

            # Check symmetry
            D_dense = D_matrix.todense()
            np.testing.assert_array_almost_equal(D_dense, D_dense.T)

    def test_mechanical_stiffness_workflow(self, sample_config):
        """Test mechanical stiffness matrix construction workflow."""
        config = sample_config

        # Setup parameters
        n_nodes = 100
        n_G = 8
        dimension = 3

        # Create mock shape functions
        phi = np.random.rand(n_G, n_nodes) * 0.1
        grad_phi = np.random.rand(n_G, n_nodes, dimension) * 0.1
        Jwei = np.ones(n_G) * (1.0 / n_G)
        indx = np.ones(n_G)

        # Step 1: Create C tensor for electrolyte
        C_tensor = mechanical_C_tensor_3d(
            config.LAMBDA_MECHANICAL_ELECTROLYTE, config.MU_ELECTROLYTE
        )

        # Check C tensor properties
        assert C_tensor.shape == (6, 6)
        # C tensor should be symmetric
        np.testing.assert_array_almost_equal(C_tensor, C_tensor.T)

        # Step 2: Create stiffness matrix
        K_matrix = mechanical_stiffness_matrix_3d_fuel_cell(
            n_nodes,
            n_G,
            Jwei,
            grad_phi,
            config.LAMBDA_MECHANICAL_ELECTROLYTE,
            config.MU_ELECTROLYTE,
            indx,
        )

        # Check stiffness matrix properties
        expected_size = n_nodes * dimension
        assert K_matrix.shape == (expected_size, expected_size)
        assert sp.issparse(K_matrix)

        # Stiffness matrix should be symmetric
        K_dense = K_matrix.todense()
        np.testing.assert_array_almost_equal(K_dense, K_dense.T, decimal=10)

        # Step 3: Create force matrix
        c_field = np.random.rand(n_nodes) * 1000  # Concentration field
        F_matrix = mechanical_force_matrix_3d(
            n_nodes,
            n_G,
            Jwei,
            grad_phi,
            phi,
            c_field,
            config.LAMBDA_MECHANICAL_ELECTROLYTE,
            config.MU_ELECTROLYTE,
            config.BETA_FUELCELL_EXPANSION_COEFFICIENT,
            indx,
        )

        # Check force matrix properties
        assert F_matrix.shape == (expected_size,)

    def test_multi_phase_system(self, sample_3d_image, sample_config):
        """Test handling of multi-phase system with different materials."""
        img_path, img_data = sample_3d_image
        config = sample_config

        # Read image to get phase distribution
        _, unique_ids, num_pixels = read_in_image(img_path, "fuel cell", 3)

        # Flatten image to get phase at each voxel
        phase_map = img_data.flatten()

        # Material properties for each phase
        material_props = {
            0: {  # Pore
                "diffusion": config.DIFFUSION_PORE,
                "lambda": 0.0,  # No mechanical properties for pore
                "mu": 0.0,
            },
            1: {  # Electrolyte
                "diffusion": config.DIFFUSION_ELECTROLYTE,
                "lambda": config.LAMBDA_MECHANICAL_ELECTROLYTE,
                "mu": config.MU_ELECTROLYTE,
            },
            2: {  # Electrode
                "diffusion": config.DIFFUSION_ELECTRODE,
                "lambda": config.LAMBDA_MECHANICAL_ELECTRODE,
                "mu": config.MU_ELECTRODE,
            },
        }

        # Verify all phases have properties defined
        for phase_id in unique_ids:
            assert phase_id in material_props
            props = material_props[phase_id]
            assert "diffusion" in props
            assert "lambda" in props
            assert "mu" in props

        # Check material property relationships
        # Diffusion: typically electrolyte > pore > electrode
        assert material_props[1]["diffusion"] > material_props[2]["diffusion"]
        assert material_props[0]["diffusion"] > material_props[2]["diffusion"]

        # Mechanical properties should be positive for solid phases
        assert material_props[1]["lambda"] > 0
        assert material_props[1]["mu"] > 0
        assert material_props[2]["lambda"] > 0
        assert material_props[2]["mu"] > 0

    def test_concentration_field_evolution(self, sample_config):
        """Test evolution of concentration field (simplified)."""
        config = sample_config

        # Setup simple 1D problem
        n_nodes = 10
        n_G = 2

        # Initial concentration field
        c_initial = np.ones(n_nodes) * config.C_BOUNDARY

        # Create simple diffusion matrix (1D)
        grad_phi = np.zeros((n_G, n_nodes, 1))
        for i in range(n_nodes - 1):
            grad_phi[0, i, 0] = -1.0 / (n_nodes - 1)
            grad_phi[0, i + 1, 0] = 1.0 / (n_nodes - 1)

        Jwei = np.ones(n_G) * 0.5
        indx = np.ones(n_G)

        D_matrix = diffusion_matrix_fuel_cell(
            n_nodes, n_G, Jwei, grad_phi, config.DIFFUSION_ELECTROLYTE, indx
        )

        # Apply boundary conditions (simplified)
        c_field = c_initial.copy()
        c_field[0] = config.C_BOUNDARY  # Left boundary
        c_field[-1] = config.C_BOUNDARY_PORE  # Right boundary

        # Check that boundary values are different (driving force)
        assert c_field[0] != c_field[-1]

        # The diffusion matrix should be constructed
        assert D_matrix is not None
        assert D_matrix.shape == (n_nodes, n_nodes)

    def test_butler_volmer_coupling(self):
        """Test coupling between Butler-Volmer equations and transport."""
        from define_buttler_volmer import i_0_complex, i_se, ocp_complex

        # Test concentration-dependent parameters
        concentrations = np.linspace(0.1, 0.9, 5)

        for c in concentrations:
            # Get concentration-dependent exchange current
            i0, di0_dc = i_0_complex(c)
            assert i0 > 0

            # Get open circuit potential
            E_eq, dE_dc = ocp_complex(c)
            assert E_eq is not None

            # Calculate current density
            p_s = 3.5  # Solid potential
            Fday = 96485
            R = 8.3145
            T = 298.15

            dibv_deta, dibv_di0, i_bv = i_se(p_s, i0, E_eq, Fday, R, T)

            # Current should depend on concentration through i0 and E_eq
            assert i_bv is not None

    def test_damage_evolution_coupling(self):
        """Test damage evolution coupling with diffusion."""
        from define_buttler_volmer import Dn_complex

        # Initial state - no damage
        x = 0.5  # Normalized concentration
        damage_initial = np.zeros(10)
        D_initial, _ = Dn_complex(x, damage_initial)

        # Evolved state - with damage
        damage_evolved = np.linspace(0, 0.5, 10)
        D_damaged, _ = Dn_complex(x, damage_evolved)

        # Diffusivity should decrease with damage
        assert np.all(D_damaged <= D_initial)

        # Check damage threshold behavior
        damage_high = np.ones(10) * 0.95  # Above 0.9 threshold
        D_high_damage, _ = Dn_complex(x, damage_high)

        # Should handle high damage gracefully
        assert D_high_damage is not None
        assert np.all(D_high_damage >= 0)  # Still non-negative

    def test_gauss_integration_consistency(self, sample_config):
        """Test consistency of Gauss integration."""
        config = sample_config

        # Test that Gauss weights sum correctly
        assert np.sum(config.WEIGHT_G_CUBE) == pytest.approx(8.0)  # 3D cube
        assert np.sum(config.WEIGHT_G_REC) == pytest.approx(4.0)  # 2D rectangle
        assert np.sum(config.WEIGHT_G_TRI) == pytest.approx(1.0)  # 2D triangle
        assert np.sum(config.WEIGHT_G_LINE) == pytest.approx(2.0)  # 1D line

        # Test that Gauss points are within reference element
        assert np.all(np.abs(config.X_G_CUBE) <= 1.0)
        assert np.all(np.abs(config.X_G_REC) <= 1.0)
        assert np.all(config.X_G_TRI >= 0) and np.all(config.X_G_TRI <= 1)
        assert np.all(np.abs(config.X_G_LINE) <= 1.0)

    def test_boundary_condition_application(self, sample_config):
        """Test boundary condition application in the system."""
        config = sample_config

        # Setup boundary nodes (simplified)
        n_boundary_nodes = 20
        n_total_nodes = 100

        # Boundary conditions
        c_left = config.C_BOUNDARY
        c_right = config.C_BOUNDARY_PORE
        v_applied = config.V_APP

        # Check that boundary conditions create driving forces
        assert c_left != c_right  # Concentration gradient
        assert v_applied > 0  # Applied voltage

        # Verify boundary condition consistency
        assert c_left > 0
        assert c_right > 0
        assert v_applied > config.E_0  # Applied voltage should overcome equilibrium

    def test_temperature_dependent_properties(self, sample_config):
        """Test temperature-dependent property calculations."""
        config = sample_config

        T = config.TEMPERATURE
        R = config.GAS_CONSTANT
        F = config.FARADAY_CONSTANT

        # Test Nernst equation factor
        nernst_factor = R * T / F
        assert nernst_factor > 0

        # At fuel cell operating temperature
        assert T > 1000  # High temperature SOFC

        # Check thermal voltage
        thermal_voltage = R * T / F
        expected_thermal = 8.3145 * 1273.2 / 96485
        assert thermal_voltage == pytest.approx(expected_thermal, rel=1e-4)

    @pytest.mark.parametrize("dimension", [2, 3])
    def test_dimension_compatibility(self, dimension):
        """Test that system handles different dimensions correctly."""
        # Setup parameters for different dimensions
        n_nodes = 50
        n_G = 4 if dimension == 2 else 8

        # Shape functions gradient dimension
        grad_phi = np.random.rand(n_G, n_nodes, dimension) * 0.1

        # Integration weights
        Jwei = np.ones(n_G) / n_G
        indx = np.ones(n_G)

        # Create diffusion matrix
        D_matrix = diffusion_matrix_fuel_cell(
            n_nodes, n_G, Jwei, grad_phi, 1.0, indx  # Unit diffusion
        )

        # Should work for both 2D and 3D
        assert D_matrix.shape == (n_nodes, n_nodes)

        if dimension == 3:
            # Create mechanical stiffness matrix (only for 3D)
            from define_mechanical_stiffness_matrix import (
                mechanical_stiffness_matrix_3d_fuel_cell,
            )

            K_matrix = mechanical_stiffness_matrix_3d_fuel_cell(
                n_nodes, n_G, Jwei, grad_phi, 100.0, 50.0, indx  # Lambda  # Mu
            )

            expected_size = n_nodes * dimension
            assert K_matrix.shape == (expected_size, expected_size)
