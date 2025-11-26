"""
Unit tests for the diffusion matrix module.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from define_diffusion_matrix_form import (
    diffusion_matrix_fuel_cell,
    diffusion_matrix_fuel_cell_distributed_point_source,
)


class TestDiffusionMatrix:
    """Test suite for diffusion matrix formulation."""

    @pytest.fixture
    def setup_basic_params(self):
        """Setup basic parameters for diffusion matrix tests."""
        # Number of nodes
        n_nodes = 10

        # Number of Gauss points
        n_G = 8

        # Create sample shape functions
        phi = np.random.rand(n_G, n_nodes)

        # Create sample shape function gradients (3D)
        grad_phi = np.random.rand(n_G, n_nodes, 3)

        # Create sample weights
        Jwei = np.ones(n_G) * 0.125  # Equal weights summing to 1

        # Domain indicator (1 for active domain)
        indx = np.ones(n_G)

        # Diffusion coefficients for different phases
        diff_coefficient = {"electrolyte": 0.035, "electrode": 1.0e-9, "pore": 1.0e-7}

        return {
            "n_nodes": n_nodes,
            "n_G": n_G,
            "phi": phi,
            "grad_phi": grad_phi,
            "Jwei": Jwei,
            "indx": indx,
            "diff_coefficient": diff_coefficient,
        }

    def test_diffusion_matrix_fuel_cell_shape(self, setup_basic_params):
        """Test that diffusion matrix has correct shape."""
        params = setup_basic_params

        # Call the function
        D_matrix = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
        )

        # Check shape
        assert D_matrix.shape == (params["n_nodes"], params["n_nodes"])

        # Check that it's a sparse matrix
        assert sp.issparse(D_matrix)

    def test_diffusion_matrix_symmetry(self, setup_basic_params):
        """Test that diffusion matrix is symmetric."""
        params = setup_basic_params

        D_matrix = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
        )

        # Convert to dense for comparison
        D_dense = D_matrix.toarray()

        # Check symmetry
        np.testing.assert_array_almost_equal(D_dense, D_dense.T)

    def test_diffusion_matrix_positive_semidefinite(self, setup_basic_params):
        """Test that diffusion matrix is positive semi-definite."""
        params = setup_basic_params

        # Create a simpler gradient for guaranteed positive semi-definiteness
        grad_phi_simple = np.random.rand(params["n_G"], params["n_nodes"], 3) * 0.1

        D_matrix = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            grad_phi_simple,
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
        )

        # Check eigenvalues
        D_dense = D_matrix.toarray()
        eigenvalues = np.linalg.eigvalsh(D_dense)

        # All eigenvalues should be non-negative (allowing small numerical errors)
        assert np.all(eigenvalues >= -1e-10)

    def test_diffusion_coefficient_scaling(self, setup_basic_params):
        """Test that matrix scales linearly with diffusion coefficient."""
        params = setup_basic_params

        # Create matrix with unit diffusion
        D_matrix_1 = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            1.0,
            params["indx"],
        )

        # Create matrix with scaled diffusion
        scale_factor = 2.5
        D_matrix_scaled = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            scale_factor,
            params["indx"],
        )

        # Check scaling
        np.testing.assert_array_almost_equal(
            D_matrix_scaled.toarray(), scale_factor * D_matrix_1.toarray()
        )

    def test_inactive_domain_points(self, setup_basic_params):
        """Test handling of inactive domain points."""
        params = setup_basic_params

        # Set some points as inactive
        indx_partial = params["indx"].copy()
        indx_partial[::2] = 0  # Every other point is inactive

        D_matrix_partial = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            indx_partial,
        )

        # Matrix should still be valid
        assert D_matrix_partial.shape == (params["n_nodes"], params["n_nodes"])

        # Check that it's still symmetric
        D_dense = D_matrix_partial.toarray()
        np.testing.assert_array_almost_equal(D_dense, D_dense.T)

    def test_distributed_point_source_shape(self, setup_basic_params):
        """Test distributed point source diffusion matrix shape."""
        params = setup_basic_params

        # Additional parameters for distributed point source
        phi_bc = np.random.rand(4, params["n_nodes"])  # Boundary shape functions
        Jwei_bc = np.ones(4) * 0.25  # Boundary weights
        indx_bc = np.ones(4)  # Boundary indicator

        D_matrix = diffusion_matrix_fuel_cell_distributed_point_source(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
            4,  # n_G_bc
            phi_bc,
            Jwei_bc,
            indx_bc,
        )

        # Check shape
        assert D_matrix.shape == (params["n_nodes"], params["n_nodes"])

        # Check that it's a sparse matrix
        assert sp.issparse(D_matrix)

    def test_distributed_point_source_symmetry(self, setup_basic_params):
        """Test that distributed point source matrix is symmetric."""
        params = setup_basic_params

        # Boundary parameters
        n_G_bc = 4
        phi_bc = np.random.rand(n_G_bc, params["n_nodes"])
        Jwei_bc = np.ones(n_G_bc) * 0.25
        indx_bc = np.ones(n_G_bc)

        D_matrix = diffusion_matrix_fuel_cell_distributed_point_source(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
            n_G_bc,
            phi_bc,
            Jwei_bc,
            indx_bc,
        )

        # Check symmetry
        D_dense = D_matrix.toarray()
        np.testing.assert_array_almost_equal(D_dense, D_dense.T)

    def test_zero_diffusion_coefficient(self, setup_basic_params):
        """Test behavior with zero diffusion coefficient."""
        params = setup_basic_params

        D_matrix = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            0.0,  # Zero diffusion
            params["indx"],
        )

        # Matrix should be zero
        assert np.allclose(D_matrix.toarray(), 0.0)

    def test_different_phase_diffusion_coefficients(self, setup_basic_params):
        """Test with different material phase diffusion coefficients."""
        params = setup_basic_params

        # Test with different phase coefficients
        for phase, coeff in params["diff_coefficient"].items():
            D_matrix = diffusion_matrix_fuel_cell(
                params["n_nodes"],
                params["n_G"],
                params["Jwei"],
                params["grad_phi"],
                coeff,
                params["indx"],
            )

            # Matrix should be valid
            assert D_matrix.shape == (params["n_nodes"], params["n_nodes"])

            # Check sparsity
            assert sp.issparse(D_matrix)

            # Check that matrix norm scales with coefficient
            matrix_norm = np.linalg.norm(D_matrix.toarray())

            # Smaller coefficients should give smaller norms
            if phase == "electrode":  # Smallest coefficient
                assert matrix_norm < 1e-6
            elif phase == "pore":  # Medium coefficient
                assert matrix_norm < 1e-4
            # electrolyte has largest coefficient

    def test_weight_normalization_effect(self, setup_basic_params):
        """Test effect of weight normalization."""
        params = setup_basic_params

        # Create unnormalized weights
        Jwei_unnorm = params["Jwei"] * 10.0

        D_matrix_norm = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            params["Jwei"],
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
        )

        D_matrix_unnorm = diffusion_matrix_fuel_cell(
            params["n_nodes"],
            params["n_G"],
            Jwei_unnorm,
            params["grad_phi"],
            params["diff_coefficient"]["electrolyte"],
            params["indx"],
        )

        # Unnormalized should be scaled version of normalized
        np.testing.assert_array_almost_equal(
            D_matrix_unnorm.toarray(), 10.0 * D_matrix_norm.toarray()
        )

    def test_gradient_dimension_consistency(self):
        """Test that gradient dimensions are handled correctly."""
        n_nodes = 5
        n_G = 3

        # Test 1D gradient
        grad_phi_1d = np.random.rand(n_G, n_nodes, 1)

        # Test 2D gradient
        grad_phi_2d = np.random.rand(n_G, n_nodes, 2)

        # Test 3D gradient
        grad_phi_3d = np.random.rand(n_G, n_nodes, 3)

        Jwei = np.ones(n_G) / n_G
        indx = np.ones(n_G)
        diff_coeff = 1.0

        for grad_phi in [grad_phi_1d, grad_phi_2d, grad_phi_3d]:
            D_matrix = diffusion_matrix_fuel_cell(
                n_nodes, n_G, Jwei, grad_phi, diff_coeff, indx
            )

            # Should produce valid matrix regardless of dimension
            assert D_matrix.shape == (n_nodes, n_nodes)
            assert sp.issparse(D_matrix)
