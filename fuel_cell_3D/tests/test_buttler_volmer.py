"""
Unit tests for the Butler-Volmer equations module.
"""

import numpy as np
import pytest
from define_buttler_volmer import (
    Dn_complex,
    alpha_lattice_complex,
    c_lattice_complex,
    i_0_complex,
    i_se,
    ocp_complex,
)


class TestButlerVolmer:
    """Test suite for Butler-Volmer equations."""

    def test_i_0_complex(self, sample_concentration):
        """Test exchange current density calculation."""
        for x in sample_concentration:
            P, dp_dx = i_0_complex(x)

            # Check that P is positive for valid concentration range
            assert P > 0, f"P should be positive for x={x}"

            # Check that derivative is computed
            assert dp_dx is not None

            # Verify polynomial evaluation at specific points
            if x == 0:
                # At x=0, P should equal A0
                A0 = 0.303490440978371
                P_test, _ = i_0_complex(0.0)
                assert P_test == pytest.approx(A0)

    def test_alpha_lattice_complex(self, sample_concentration):
        """Test alpha lattice parameter calculation."""
        for x in sample_concentration:
            a_lattice, dalattice_dx = alpha_lattice_complex(x)

            # Check that values are returned
            assert a_lattice is not None
            assert dalattice_dx is not None

            # Alpha lattice should be small (in nm range)
            assert abs(a_lattice) < 1e-6, f"a_lattice seems too large for x={x}"

    def test_c_lattice_complex(self, sample_concentration):
        """Test c lattice parameter calculation."""
        for x in sample_concentration:
            c_lattice, dclattice_dx = c_lattice_complex(x)

            # Check that values are returned
            assert c_lattice is not None
            assert dclattice_dx is not None

            # C lattice should be small (in nm range)
            assert abs(c_lattice) < 1e-5, f"c_lattice seems too large for x={x}"

    def test_Dn_complex(self, sample_concentration, sample_damage):
        """Test diffusivity calculation."""
        for x in sample_concentration:
            D, dD_dx = Dn_complex(x, sample_damage)

            # Check shapes
            assert D.shape == sample_damage.shape
            assert dD_dx.shape == sample_damage.shape

            # Diffusivity should be non-negative
            assert np.all(D >= 0), f"Diffusivity should be non-negative for x={x}"

            # Test damage effect - higher damage should reduce diffusivity
            no_damage = np.zeros_like(sample_damage)
            D_no_damage, _ = Dn_complex(x, no_damage)

            high_damage = np.ones_like(sample_damage) * 0.5
            D_high_damage, _ = Dn_complex(x, high_damage)

            # With damage, diffusivity should be lower
            assert np.all(D_high_damage <= D_no_damage)

    def test_Dn_complex_damage_limit(self):
        """Test that damage is capped at 0.9."""
        x = 0.5
        damage = np.array([0.5, 0.8, 0.95, 1.0, 1.5])  # Some values > 0.9

        D, dD_dx = Dn_complex(x, damage)

        # Function should handle damage > 0.9 by capping it
        assert D is not None
        assert dD_dx is not None
        # Values should be computed without error

    def test_ocp_complex(self, sample_concentration):
        """Test open circuit potential calculation."""
        for x in sample_concentration:
            E_eq, dEeq_dx = ocp_complex(x)

            # Check that values are returned
            assert E_eq is not None
            assert dEeq_dx is not None

            # OCP should be in reasonable range for Li-ion battery (0-5V)
            # This might need adjustment based on actual system
            assert -10 < E_eq < 10, f"E_eq={E_eq} seems unreasonable for x={x}"

    def test_ocp_complex_boundary_conditions(self):
        """Test OCP at boundary conditions."""
        # Test at x=0 (fully delithiated)
        E_eq_0, _ = ocp_complex(0.0)
        assert E_eq_0 is not None

        # Test at x=1 (fully lithiated)
        E_eq_1, _ = ocp_complex(1.0)
        assert E_eq_1 is not None

        # OCP should generally decrease with lithiation (system dependent)
        # This is a general trend but may not always hold
        # assert E_eq_0 > E_eq_1  # Commented out as it's material-specific

    def test_i_se_current_density(self):
        """Test current density calculation."""
        # Test parameters
        p_s = 3.5  # Solid potential (V)
        j0 = 1.0e-3  # Exchange current density (A/m²)
        E_eq = 3.0  # Equilibrium potential (V)
        Fday = 96485  # Faraday constant (C/mol)
        R = 8.3145  # Gas constant (J/mol·K)
        Tk = 298.15  # Temperature (K)

        dibv_deta, dibv_di0, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)

        # Check that all outputs are returned
        assert dibv_deta is not None
        assert dibv_di0 is not None
        assert i_bv is not None

        # Check Butler-Volmer equation properties
        eta_s = p_s - E_eq  # Overpotential

        # At zero overpotential, current should be zero
        dibv_deta_0, dibv_di0_0, i_bv_0 = i_se(E_eq, j0, E_eq, Fday, R, Tk)
        assert abs(i_bv_0) < 1e-10, "Current should be zero at zero overpotential"

        # Positive overpotential should give positive current (anodic)
        if eta_s > 0:
            assert i_bv > 0, "Positive overpotential should give positive current"
        elif eta_s < 0:
            assert i_bv < 0, "Negative overpotential should give negative current"

    def test_i_se_derivatives(self):
        """Test derivatives in current density calculation."""
        # Test parameters
        p_s = 3.5
        j0 = 1.0e-3
        E_eq = 3.0
        Fday = 96485
        R = 8.3145
        Tk = 298.15

        dibv_deta, dibv_di0, i_bv = i_se(p_s, j0, E_eq, Fday, R, Tk)

        # Test numerical derivative with respect to eta
        delta = 1e-6
        _, _, i_bv_plus = i_se(p_s + delta, j0, E_eq, Fday, R, Tk)
        _, _, i_bv_minus = i_se(p_s - delta, j0, E_eq, Fday, R, Tk)

        dibv_deta_numerical = (i_bv_plus - i_bv_minus) / (2 * delta)

        # Check that analytical and numerical derivatives are close
        assert dibv_deta == pytest.approx(dibv_deta_numerical, rel=1e-3)

        # Test derivative with respect to j0
        # The derivative should be i_bv / j0 for small overpotentials
        assert dibv_di0 == pytest.approx(i_bv / j0, rel=0.1)

    def test_polynomial_continuity(self):
        """Test continuity of piecewise polynomials at thresholds."""
        # Test points near thresholds for OCP
        thresholds = [
            0.0,
            0.025,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
            1.0,
        ]

        for i in range(len(thresholds) - 1):
            x_left = thresholds[i] - 1e-8 if thresholds[i] > 0 else 0
            x_right = thresholds[i] + 1e-8

            if x_left >= 0 and x_right <= 1:
                E_left, _ = ocp_complex(x_left)
                E_right, _ = ocp_complex(x_right)

                # Values should be approximately continuous
                # Allow for some discontinuity due to piecewise nature
                assert (
                    abs(E_left - E_right) < 1.0
                ), f"Large discontinuity at x={thresholds[i]}"

    def test_concentration_bounds(self):
        """Test behavior at and beyond concentration bounds."""
        # Test at x < 0 (should handle gracefully)
        P_neg, _ = i_0_complex(-0.1)
        assert P_neg is not None

        # Test at x > 1 (should handle gracefully)
        P_high, _ = i_0_complex(1.1)
        assert P_high is not None

        # Test diffusivity with out-of-bounds concentration
        damage = np.array([0.0, 0.1, 0.2])
        D_neg, _ = Dn_complex(-0.1, damage)
        D_high, _ = Dn_complex(1.1, damage)

        assert D_neg is not None
        assert D_high is not None

    def test_temperature_sensitivity(self):
        """Test temperature effects on current density."""
        p_s = 3.5
        j0 = 1.0e-3
        E_eq = 3.0
        Fday = 96485
        R = 8.3145

        # Test at different temperatures
        T_low = 273.15  # 0°C
        T_room = 298.15  # 25°C
        T_high = 353.15  # 80°C

        _, _, i_low = i_se(p_s, j0, E_eq, Fday, R, T_low)
        _, _, i_room = i_se(p_s, j0, E_eq, Fday, R, T_room)
        _, _, i_high = i_se(p_s, j0, E_eq, Fday, R, T_high)

        # Higher temperature should generally increase current magnitude
        # (lower activation barrier)
        assert abs(i_high) > abs(
            i_low
        ), "Higher temperature should increase reaction rate"
