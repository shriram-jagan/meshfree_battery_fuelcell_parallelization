from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """Material Properties"""

    Fday = 9.6485e4  # Faraday constant
    R = 8.3145e0  # gas constant
    Tk = 3.0515e2  # temperature in K

    c_max = 49600.0  # maximum concentration

    k_con = 10.0  # conductivity

    Dx_div_Dy = 100.0

    j_applied = -15.0  # j_applied

    E = 138.87e9  # Youngs modulus (Pa)
    nu = 0.3  # Poisson ratio
    mu = E / 2 / (1 + nu)  # lamme constants

    k_i = 0.0125
    k_f = 0.015

    def lambda_mechanical(self) -> float:
        return self.E * self.nu / (1.0 + self.nu) / (1.0 - 2.0 * self.nu)
