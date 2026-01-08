"""
Mechanical stiffness matrix assembly for 3D linear elasticity.

This module assembles the mechanical stiffness matrix for 3D elasticity
problems in solid oxide fuel cells. It includes:

- Construction of the 6x6 elasticity tensor (Voigt notation)
- Rotation for anisotropic materials
- Damage model degradation
- Full 3x3 block stiffness matrix assembly

The mechanical stiffness matrix relates nodal displacements to forces:

    K_mech @ u = f

where u = [u_x, u_y, u_z] and K_mech is a 3x3 block matrix.
"""

from typing import Tuple

import config
from common import bmat, csr_array, np, sp


def mechanical_C_tensor_3d(
    num_gauss_points_in_domain: int,
    D_damage: np.ndarray,
    lambda_mechanical: np.ndarray,
    mu: np.ndarray,
    gauss_angle: np.ndarray,
    gauss_rotation_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the 3D elasticity tensor with rotation and damage.

    Computes the 6x6 elasticity tensor in Voigt notation for each Gauss point,
    including material rotation and damage degradation.

    The elasticity tensor relates stress to strain:

        σ = C : (ε - ε*)

    where ε* is the chemical expansion strain.

    Parameters
    ----------
    num_gauss_points_in_domain : int
        Number of Gauss points in the domain.
    D_damage : np.ndarray
        Damage variable at each Gauss point, shape (n_gauss, 1).
        Value 0 = undamaged, 1 = fully damaged.
    lambda_mechanical : np.ndarray
        First Lamé constant (λ) at each Gauss point.
    mu : np.ndarray
        Second Lamé constant (shear modulus, μ) at each Gauss point.
    gauss_angle : np.ndarray
        Rotation angle at each Gauss point for anisotropic materials.
    gauss_rotation_axis : np.ndarray
        Rotation axis vectors, shape (n_gauss, 3).

    Returns
    -------
    C : np.ndarray
        Elasticity tensor, shape (6, 6, n_gauss, 1).
        Components are in Voigt notation: [11, 22, 33, 23, 13, 12].
    T_c : np.ndarray
        Rotation transformation matrix, shape (6, 6, n_gauss).
    """
    D_damage[D_damage > 0.9] = 0.9  # c = max(1-c_no_damage, 0.1)

    C11_ini = lambda_mechanical + 2 * mu  # c11 before rotation
    C22_ini = lambda_mechanical + 2 * mu
    C33_ini = lambda_mechanical + 2 * mu
    C44_ini = mu
    C55_ini = mu
    C66_ini = mu
    C12_ini = lambda_mechanical
    C13_ini = lambda_mechanical
    C23_ini = lambda_mechanical
    C21_ini = C12_ini
    C31_ini = C13_ini
    C32_ini = C23_ini

    if len(np.shape(lambda_mechanical)) == 0:
        C14_ini = 0
        C15_ini = 0
        C16_ini = 0

        C24_ini = 0
        C25_ini = 0
        C26_ini = 0

        C34_ini = 0
        C35_ini = 0
        C36_ini = 0

        C41_ini = 0
        C42_ini = 0
        C43_ini = 0
        C45_ini = 0
        C46_ini = 0

        C51_ini = 0
        C52_ini = 0
        C53_ini = 0
        C54_ini = 0
        C56_ini = 0

        C61_ini = 0
        C62_ini = 0
        C63_ini = 0
        C64_ini = 0
        C65_ini = 0

    else:
        C14_ini = np.zeros(num_gauss_points_in_domain)
        C15_ini = np.zeros(num_gauss_points_in_domain)
        C16_ini = np.zeros(num_gauss_points_in_domain)

        C24_ini = np.zeros(num_gauss_points_in_domain)
        C25_ini = np.zeros(num_gauss_points_in_domain)
        C26_ini = np.zeros(num_gauss_points_in_domain)

        C34_ini = np.zeros(num_gauss_points_in_domain)
        C35_ini = np.zeros(num_gauss_points_in_domain)
        C36_ini = np.zeros(num_gauss_points_in_domain)

        C41_ini = np.zeros(num_gauss_points_in_domain)
        C42_ini = np.zeros(num_gauss_points_in_domain)
        C43_ini = np.zeros(num_gauss_points_in_domain)
        C45_ini = np.zeros(num_gauss_points_in_domain)
        C46_ini = np.zeros(num_gauss_points_in_domain)

        C51_ini = np.zeros(num_gauss_points_in_domain)
        C52_ini = np.zeros(num_gauss_points_in_domain)
        C53_ini = np.zeros(num_gauss_points_in_domain)
        C54_ini = np.zeros(num_gauss_points_in_domain)
        C56_ini = np.zeros(num_gauss_points_in_domain)

        C61_ini = np.zeros(num_gauss_points_in_domain)
        C62_ini = np.zeros(num_gauss_points_in_domain)
        C63_ini = np.zeros(num_gauss_points_in_domain)
        C64_ini = np.zeros(num_gauss_points_in_domain)
        C65_ini = np.zeros(num_gauss_points_in_domain)

        C_ini = np.array(
            [
                [C11_ini, C12_ini, C13_ini, C14_ini, C15_ini, C16_ini],
                [C21_ini, C22_ini, C23_ini, C24_ini, C25_ini, C26_ini],
                [C31_ini, C32_ini, C33_ini, C34_ini, C35_ini, C36_ini],
                [C41_ini, C42_ini, C43_ini, C44_ini, C45_ini, C46_ini],
                [C51_ini, C52_ini, C53_ini, C54_ini, C55_ini, C56_ini],
                [C61_ini, C62_ini, C63_ini, C64_ini, C65_ini, C66_ini],
            ]
        )

    R11_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 0] ** 2 * (
        1 - np.cos(gauss_angle)
    )
    R12_c = gauss_rotation_axis[:, 0] * gauss_rotation_axis[:, 1] * (
        1 - np.cos(gauss_angle)
    ) - gauss_rotation_axis[:, 2] * np.sin(gauss_angle)
    R13_c = gauss_rotation_axis[:, 0] * gauss_rotation_axis[:, 2] * (
        1 - np.cos(gauss_angle)
    ) + gauss_rotation_axis[:, 1] * np.sin(gauss_angle)

    R21_c = gauss_rotation_axis[:, 0] * gauss_rotation_axis[:, 1] * (
        1 - np.cos(gauss_angle)
    ) + gauss_rotation_axis[:, 2] * np.sin(gauss_angle)
    R22_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 1] ** 2 * (
        1 - np.cos(gauss_angle)
    )
    R23_c = gauss_rotation_axis[:, 2] * gauss_rotation_axis[:, 1] * (
        1 - np.cos(gauss_angle)
    ) - gauss_rotation_axis[:, 0] * np.sin(gauss_angle)

    R31_c = gauss_rotation_axis[:, 2] * gauss_rotation_axis[:, 0] * (
        1 - np.cos(gauss_angle)
    ) - gauss_rotation_axis[:, 1] * np.sin(gauss_angle)
    R32_c = gauss_rotation_axis[:, 2] * gauss_rotation_axis[:, 1] * (
        1 - np.cos(gauss_angle)
    ) + gauss_rotation_axis[:, 0] * np.sin(gauss_angle)
    R33_c = np.cos(gauss_angle) + gauss_rotation_axis[:, 2] ** 2 * (
        1 - np.cos(gauss_angle)
    )

    T11_c = R11_c**2
    T12_c = R12_c**2
    T13_c = R13_c**2
    T14_c = 2 * R12_c * R13_c
    T15_c = 2 * R11_c * R13_c
    T16_c = 2 * R11_c * R12_c

    T21_c = R21_c**2
    T22_c = R22_c**2
    T23_c = R23_c**2
    T24_c = 2 * R22_c * R23_c
    T25_c = 2 * R21_c * R23_c
    T26_c = 2 * R21_c * R22_c

    T31_c = R31_c**2
    T32_c = R32_c**2
    T33_c = R33_c**2
    T34_c = 2 * R32_c * R33_c
    T35_c = 2 * R31_c * R33_c
    T36_c = 2 * R31_c * R32_c

    T41_c = R21_c * R31_c
    T42_c = R22_c * R32_c
    T43_c = R23_c * R33_c
    T44_c = R22_c * R33_c + R23_c * R32_c
    T45_c = R21_c * R33_c + R23_c * R31_c
    T46_c = R21_c * R32_c + R22_c * R31_c

    T51_c = R11_c * R31_c
    T52_c = R12_c * R32_c
    T53_c = R13_c * R33_c
    T54_c = R12_c * R33_c + R13_c * R32_c
    T55_c = R11_c * R33_c + R13_c * R31_c
    T56_c = R11_c * R32_c + R12_c * R31_c

    T61_c = R11_c * R21_c
    T62_c = R12_c * R22_c
    T63_c = R13_c * R23_c
    T64_c = R12_c * R23_c + R13_c * R22_c
    T65_c = R11_c * R23_c + R13_c * R21_c
    T66_c = R11_c * R22_c + R12_c * R21_c

    T_c = np.array(
        [
            [T11_c, T12_c, T13_c, T14_c, T15_c, T16_c],
            [T21_c, T22_c, T23_c, T24_c, T25_c, T26_c],
            [T31_c, T32_c, T33_c, T34_c, T35_c, T36_c],
            [T41_c, T42_c, T43_c, T44_c, T45_c, T46_c],
            [T51_c, T52_c, T53_c, T54_c, T55_c, T56_c],
            [T61_c, T62_c, T63_c, T64_c, T65_c, T66_c],
        ]
    )

    # stiff tensor after rotation:
    if len(np.shape(lambda_mechanical)) == 0:
        C_mechanical = np.einsum(
            "imk,mjk->ijk",
            T_c,
            np.einsum("im,mjk->ijk", C_ini, np.transpose(T_c, (1, 0, 2))),
        )  # stiffness without damage
    else:
        C_mechanical = np.einsum(
            "imk,mjk->ijk",
            T_c,
            np.einsum("imk,mjk->ijk", C_ini, np.transpose(T_c, (1, 0, 2))),
        )  # stiffness without damage

    C11 = C_mechanical[0, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C12 = C_mechanical[0, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C13 = C_mechanical[0, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C14 = C_mechanical[0, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C15 = C_mechanical[0, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C16 = C_mechanical[0, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C21 = C_mechanical[1, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C22 = C_mechanical[1, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C23 = C_mechanical[1, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C24 = C_mechanical[1, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C25 = C_mechanical[1, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C26 = C_mechanical[1, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C31 = C_mechanical[2, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C32 = C_mechanical[2, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C33 = C_mechanical[2, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C34 = C_mechanical[2, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C35 = C_mechanical[2, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C36 = C_mechanical[2, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C41 = C_mechanical[3, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C42 = C_mechanical[3, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C43 = C_mechanical[3, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C44 = C_mechanical[3, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C45 = C_mechanical[3, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C46 = C_mechanical[3, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C51 = C_mechanical[4, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C52 = C_mechanical[4, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C53 = C_mechanical[4, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C54 = C_mechanical[4, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C55 = C_mechanical[4, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C56 = C_mechanical[4, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C61 = C_mechanical[5, 0].reshape(num_gauss_points_in_domain, 1) * (
        1 - D_damage
    )  # shape: number of gauss points
    C62 = C_mechanical[5, 1].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C63 = C_mechanical[5, 2].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C64 = C_mechanical[5, 3].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C65 = C_mechanical[5, 4].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)
    C66 = C_mechanical[5, 5].reshape(num_gauss_points_in_domain, 1) * (1 - D_damage)

    C = np.array(
        [
            [C11, C12, C13, C14, C15, C16],
            [C21, C22, C23, C24, C25, C26],
            [C31, C32, C33, C34, C35, C36],
            [C41, C42, C43, C44, C45, C46],
            [C51, C52, C53, C54, C55, C56],
            [C61, C62, C63, C64, C65, C66],
        ]
    )

    return C, T_c


def mechanical_stiffness_matrix_3d_fuel_cell(
    C: np.ndarray,
    num_gauss_points_in_domain: int,
    grad_shape_func_x_times_det_J_time_weight: csr_array,
    grad_shape_func_x: csr_array,
    grad_shape_func_y_times_det_J_time_weight: csr_array,
    grad_shape_func_y: csr_array,
    grad_shape_func_z_times_det_J_time_weight: csr_array,
    grad_shape_func_z: csr_array,
    beta_Nitsche: np.ndarray,
    shape_func_fixed_point: csr_array,
    shape_func_times_det_J_time_weight_fixed_point: csr_array,
    grad_shape_func_x_fixed_point: csr_array,
    grad_shape_func_x_times_det_J_time_weight_fixed_point: csr_array,
    grad_shape_func_y_fixed_point: csr_array,
    grad_shape_func_y_times_det_J_time_weight_fixed_point: csr_array,
    grad_shape_func_z_fixed_point: csr_array,
    grad_shape_func_z_times_det_J_time_weight_fixed_point: csr_array,
    normal_vector_x_electrolyte: np.ndarray,
    normal_vector_y_electrolyte: np.ndarray,
    normal_vector_z_electrolyte: np.ndarray,
    shape_func_b_electrolyte: csr_array,
    shape_func_b_times_det_J_b_time_weight_electrolyte: csr_array,
    grad_shape_func_b_x_electrolyte: csr_array,
    grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte: csr_array,
    grad_shape_func_b_y_electrolyte: csr_array,
    grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte: csr_array,
    grad_shape_func_b_z_electrolyte: csr_array,
    grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte: csr_array,
) -> csr_array:
    """
    Assemble the 3D mechanical stiffness matrix for fuel cell simulation.

    Constructs the full 3×3 block stiffness matrix for 3D linear elasticity
    using RKPM shape functions. Includes both domain integrals and boundary
    terms (Nitsche's method for fixed displacement BCs).

    The stiffness matrix structure:

        K_mech = | K11  K12  K13 |
                 | K21  K22  K23 |
                 | K31  K32  K33 |

    where each Kij block is (n_nodes × n_nodes) and couples displacement
    components i and j.

    Parameters
    ----------
    C : np.ndarray
        Elasticity tensor, shape (6, 6, n_gauss, 1).
    num_gauss_points_in_domain : int
        Number of Gauss points in the domain.
    grad_shape_func_x, grad_shape_func_y, grad_shape_func_z : csr_array
        Shape function gradients in domain, each (n_gauss, n_nodes).
    grad_shape_func_*_times_det_J_time_weight : csr_array
        Weighted gradient matrices for integration.
    beta_Nitsche : np.ndarray
        Nitsche penalty parameter at boundary points.
    shape_func_fixed_point : csr_array
        Shape functions at fixed displacement points.
    normal_vector_*_electrolyte : np.ndarray
        Normal vector components at boundary Gauss points.
    shape_func_b_electrolyte : csr_array
        Shape functions at boundary Gauss points.
    grad_shape_func_b_*_electrolyte : csr_array
        Gradient matrices at boundary Gauss points.

    Returns
    -------
    K_mechanical : csr_array
        Mechanical stiffness matrix, shape (3*n_nodes, 3*n_nodes).
        Ordering: [u_x nodes, u_y nodes, u_z nodes].
    """

    num_gauss_points_on_boundary = np.shape(shape_func_b_electrolyte.todense())[0]
    num_gauss_points_on_fixed_line = np.shape(shape_func_fixed_point.todense())[0]

    C11 = C[0, 0].reshape(
        num_gauss_points_in_domain,
    )
    C12 = C[0, 1].reshape(
        num_gauss_points_in_domain,
    )
    C13 = C[0, 2].reshape(
        num_gauss_points_in_domain,
    )
    C14 = C[0, 3].reshape(
        num_gauss_points_in_domain,
    )
    C15 = C[0, 4].reshape(
        num_gauss_points_in_domain,
    )
    C16 = C[0, 5].reshape(
        num_gauss_points_in_domain,
    )

    C21 = C[1, 0].reshape(
        num_gauss_points_in_domain,
    )
    C22 = C[1, 1].reshape(
        num_gauss_points_in_domain,
    )
    C23 = C[1, 2].reshape(
        num_gauss_points_in_domain,
    )
    C24 = C[1, 3].reshape(
        num_gauss_points_in_domain,
    )
    C25 = C[1, 4].reshape(
        num_gauss_points_in_domain,
    )
    C26 = C[1, 5].reshape(
        num_gauss_points_in_domain,
    )

    C31 = C[2, 0].reshape(
        num_gauss_points_in_domain,
    )
    C32 = C[2, 1].reshape(
        num_gauss_points_in_domain,
    )
    C33 = C[2, 2].reshape(
        num_gauss_points_in_domain,
    )
    C34 = C[2, 3].reshape(
        num_gauss_points_in_domain,
    )
    C35 = C[2, 4].reshape(
        num_gauss_points_in_domain,
    )
    C36 = C[2, 5].reshape(
        num_gauss_points_in_domain,
    )

    C41 = C[3, 0].reshape(
        num_gauss_points_in_domain,
    )
    C42 = C[3, 1].reshape(
        num_gauss_points_in_domain,
    )
    C43 = C[3, 2].reshape(
        num_gauss_points_in_domain,
    )
    C44 = C[3, 3].reshape(
        num_gauss_points_in_domain,
    )
    C45 = C[3, 4].reshape(
        num_gauss_points_in_domain,
    )
    C46 = C[3, 5].reshape(
        num_gauss_points_in_domain,
    )

    C51 = C[4, 0].reshape(
        num_gauss_points_in_domain,
    )
    C52 = C[4, 1].reshape(
        num_gauss_points_in_domain,
    )
    C53 = C[4, 2].reshape(
        num_gauss_points_in_domain,
    )
    C54 = C[4, 3].reshape(
        num_gauss_points_in_domain,
    )
    C55 = C[4, 4].reshape(
        num_gauss_points_in_domain,
    )
    C56 = C[4, 5].reshape(
        num_gauss_points_in_domain,
    )

    C61 = C[5, 0].reshape(
        num_gauss_points_in_domain,
    )
    C62 = C[5, 1].reshape(
        num_gauss_points_in_domain,
    )
    C63 = C[5, 2].reshape(
        num_gauss_points_in_domain,
    )
    C64 = C[5, 3].reshape(
        num_gauss_points_in_domain,
    )
    C65 = C[5, 4].reshape(
        num_gauss_points_in_domain,
    )
    C66 = C[5, 5].reshape(
        num_gauss_points_in_domain,
    )

    # C11, C22, C33, C44, C55, C66, C12, C21, C13, C31, C23 and C32
    # are non-zero while the rest are zero and doesn't need to be added

    K11_mechanical_doamin_int = (
        (
            sp.dia_array(
                (C11.reshape(1, -1), 0), shape=(len(C11), len(C11)), dtype=C11.dtype
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight)
        ).T
        @ grad_shape_func_x
        + (
            sp.dia_array(
                (C44.reshape(1, -1), 0), shape=(len(C44), len(C44)), dtype=C44.dtype
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight)
        ).T
        @ grad_shape_func_y
        + (
            sp.dia_array(
                (C55.reshape(1, -1), 0), shape=(len(C55), len(C55)), dtype=C55.dtype
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight)
        ).T
        @ grad_shape_func_z
    )

    K12_mechanical_doamin_int = (
        sp.dia_array(
            (C12.reshape(1, -1), 0), shape=(len(C12), len(C12)), dtype=C12.dtype
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight)
    ).T @ grad_shape_func_y + (
        sp.dia_array(
            (C44.reshape(1, -1), 0), shape=(len(C44), len(C44)), dtype=C44.dtype
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight)
    ).T @ grad_shape_func_x

    K13_mechanical_doamin_int = (
        sp.dia_array(
            (C13.reshape(1, -1), 0), shape=(len(C13), len(C13)), dtype=C13.dtype
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight)
    ).T @ grad_shape_func_z + (
        sp.dia_array(
            (C55.reshape(1, -1), 0), shape=(len(C55), len(C55)), dtype=C55.dtype
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight)
    ).T @ grad_shape_func_x

    K21_mechanical_doamin_int = (
        sp.dia_array(
            (C21.reshape(1, -1), 0), shape=(len(C21), len(C21)), dtype=C21.dtype
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight)
    ).T @ grad_shape_func_x + (
        sp.dia_array(
            (C44.reshape(1, -1), 0), shape=(len(C44), len(C44)), dtype=C44.dtype
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight)
    ).T @ grad_shape_func_y

    K22_mechanical_doamin_int = (
        (
            sp.dia_array(
                (C22.reshape(1, -1), 0), shape=(len(C22), len(C22)), dtype=C22.dtype
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight)
        ).T
        @ grad_shape_func_y
        + (
            sp.dia_array(
                (C44.reshape(1, -1), 0), shape=(len(C44), len(C44)), dtype=C44.dtype
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight)
        ).T
        @ grad_shape_func_x
        + (
            sp.dia_array(
                (C66.reshape(1, -1), 0), shape=(len(C66), len(C66)), dtype=C66.dtype
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight)
        ).T
        @ grad_shape_func_z
    )

    K23_mechanical_doamin_int = (
        sp.dia_array(
            (C23.reshape(1, -1), 0), shape=(len(C23), len(C23)), dtype=C23.dtype
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight)
    ).T @ grad_shape_func_z + (
        sp.dia_array(
            (C66.reshape(1, -1), 0), shape=(len(C66), len(C66)), dtype=C66.dtype
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight)
    ).T @ grad_shape_func_y

    K31_mechanical_doamin_int = (
        sp.dia_array(
            (C31.reshape(1, -1), 0), shape=(len(C31), len(C31)), dtype=C31.dtype
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight)
    ).T @ grad_shape_func_x + (
        sp.dia_array(
            (C55.reshape(1, -1), 0), shape=(len(C55), len(C55)), dtype=C55.dtype
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight)
    ).T @ grad_shape_func_z

    K32_mechanical_doamin_int = (
        sp.dia_array(
            (C32.reshape(1, -1), 0), shape=(len(C32), len(C32)), dtype=C32.dtype
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight)
    ).T @ grad_shape_func_y + (
        sp.dia_array(
            (C66.reshape(1, -1), 0), shape=(len(C66), len(C66)), dtype=C66.dtype
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight)
    ).T @ grad_shape_func_z

    K33_mechanical_doamin_int = (
        (
            sp.dia_array(
                (C33.reshape(1, -1), 0), shape=(len(C33), len(C33)), dtype=C33.dtype
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight)
        ).T
        @ grad_shape_func_z
        + (
            sp.dia_array(
                (C55.reshape(1, -1), 0), shape=(len(C55), len(C55)), dtype=C55.dtype
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight)
        ).T
        @ grad_shape_func_x
        + (
            sp.dia_array(
                (C66.reshape(1, -1), 0), shape=(len(C66), len(C66)), dtype=C66.dtype
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight)
        ).T
        @ grad_shape_func_y
    )

    K11_mechanical_line_int = (
        (
            shape_func_times_det_J_time_weight_fixed_point
            @ sp.dia_array(
                (beta_Nitsche.reshape(1, -1), 0),
                shape=(len(beta_Nitsche), len(beta_Nitsche)),
                dtype=beta_Nitsche.dtype,
            ).tocsr()
        ).T
        @ shape_func_fixed_point
        - (
            sp.dia_array(
                (C11[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C11[:num_gauss_points_on_fixed_line]),
                    len(C11[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C11[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte))
        - (
            sp.dia_array(
                (C44[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C44[:num_gauss_points_on_fixed_line]),
                    len(C44[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C44[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_y_electrolyte))
        - (
            sp.dia_array(
                (C55[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C55[:num_gauss_points_on_fixed_line]),
                    len(C55[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C55[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_z_electrolyte))
    )

    # TODO:
    # SJ Wed Jan  7 04:45:19 PM PST 2026

    K12_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C44[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C44[:num_gauss_points_on_fixed_line]),
                len(C44[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C44[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) - (
        sp.dia_array(
            (C21[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C21[:num_gauss_points_on_fixed_line]),
                len(C21[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C21[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_y_electrolyte)
    )

    K12_mechanical_boundary_int = (
        -1
        * sp.dia_array(
            (C44[:num_gauss_points_on_boundary].reshape(1, -1), 0),
            shape=(
                len(C44[:num_gauss_points_on_boundary]),
                len(C44[:num_gauss_points_on_boundary]),
            ),
            dtype=C44[:num_gauss_points_on_boundary].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)
    ).T @ (shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte)) - (
        sp.dia_array(
            (C21[:num_gauss_points_on_boundary].reshape(1, -1), 0),
            shape=(
                len(C21[:num_gauss_points_on_boundary]),
                len(C21[:num_gauss_points_on_boundary]),
            ),
            dtype=C21[:num_gauss_points_on_boundary].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)
    ).T @ (
        shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)
    )

    K13_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C55[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C55[:num_gauss_points_on_fixed_line]),
                len(C55[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C55[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) - (
        sp.dia_array(
            (C31[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C31[:num_gauss_points_on_fixed_line]),
                len(C31[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C31[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_z_electrolyte)
    )

    K21_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C12[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C12[:num_gauss_points_on_fixed_line]),
                len(C12[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C12[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) - (
        sp.dia_array(
            (C44[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C44[:num_gauss_points_on_fixed_line]),
                len(C44[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C44[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_y_electrolyte)
    )

    K22_mechanical_line_int = (
        (
            shape_func_times_det_J_time_weight_fixed_point
            @ sp.dia_array(
                (beta_Nitsche.reshape(1, -1), 0),
                shape=(len(beta_Nitsche), len(beta_Nitsche)),
                dtype=beta_Nitsche.dtype,
            ).tocsr()
        ).T
        @ shape_func_fixed_point
        - (
            sp.dia_array(
                (C44[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C44[:num_gauss_points_on_fixed_line]),
                    len(C44[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C44[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte))
        - (
            sp.dia_array(
                (C22[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C22[:num_gauss_points_on_fixed_line]),
                    len(C22[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C22[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_y_electrolyte))
        - (
            sp.dia_array(
                (C66[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C66[:num_gauss_points_on_fixed_line]),
                    len(C66[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C66[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_z_electrolyte))
    )

    K22_mechanical_boundary_int = (
        (
            shape_func_b_times_det_J_b_time_weight_electrolyte
            @ sp.dia_array(
                (beta_Nitsche.reshape(1, -1), 0),
                shape=(len(beta_Nitsche), len(beta_Nitsche)),
                dtype=beta_Nitsche.dtype,
            ).tocsr()
        ).T
        @ shape_func_b_electrolyte
        - (
            sp.dia_array(
                (C44[:num_gauss_points_on_boundary].reshape(1, -1), 0),
                shape=(
                    len(C44[:num_gauss_points_on_boundary]),
                    len(C44[:num_gauss_points_on_boundary]),
                ),
                dtype=C44[:num_gauss_points_on_boundary].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_b_x_times_det_J_b_time_weight_electrolyte)
        ).T
        @ (shape_func_b_electrolyte.multiply(normal_vector_x_electrolyte))
        - (
            sp.dia_array(
                (C22[:num_gauss_points_on_boundary].reshape(1, -1), 0),
                shape=(
                    len(C22[:num_gauss_points_on_boundary]),
                    len(C22[:num_gauss_points_on_boundary]),
                ),
                dtype=C22[:num_gauss_points_on_boundary].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)
        ).T
        @ (shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte))
        - (
            sp.dia_array(
                (C66[:num_gauss_points_on_boundary].reshape(1, -1), 0),
                shape=(
                    len(C66[:num_gauss_points_on_boundary]),
                    len(C66[:num_gauss_points_on_boundary]),
                ),
                dtype=C66[:num_gauss_points_on_boundary].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)
        ).T
        @ (shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte))
    )

    K23_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C66[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C66[:num_gauss_points_on_fixed_line]),
                len(C66[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C66[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) - (
        sp.dia_array(
            (C32[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C32[:num_gauss_points_on_fixed_line]),
                len(C32[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C32[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_z_electrolyte)
    )

    K31_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C13[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C13[:num_gauss_points_on_fixed_line]),
                len(C13[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C13[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte)) - (
        sp.dia_array(
            (C55[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C55[:num_gauss_points_on_fixed_line]),
                len(C55[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C55[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_z_electrolyte)
    )

    K32_mechanical_line_int = (
        -1
        * sp.dia_array(
            (C23[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C23[:num_gauss_points_on_fixed_line]),
                len(C23[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C23[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
    ).T @ (shape_func_fixed_point.multiply(normal_vector_y_electrolyte)) - (
        sp.dia_array(
            (C66[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
            shape=(
                len(C66[:num_gauss_points_on_fixed_line]),
                len(C66[:num_gauss_points_on_fixed_line]),
            ),
            dtype=C66[:num_gauss_points_on_fixed_line].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
    ).T @ (
        shape_func_fixed_point.multiply(normal_vector_z_electrolyte)
    )

    K32_mechanical_boundary_int = (
        -1
        * sp.dia_array(
            (C23[:num_gauss_points_on_boundary].reshape(1, -1), 0),
            shape=(
                len(C23[:num_gauss_points_on_boundary]),
                len(C23[:num_gauss_points_on_boundary]),
            ),
            dtype=C23[:num_gauss_points_on_boundary].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_b_z_times_det_J_b_time_weight_electrolyte)
    ).T @ (shape_func_b_electrolyte.multiply(normal_vector_y_electrolyte)) - (
        sp.dia_array(
            (C66[:num_gauss_points_on_boundary].reshape(1, -1), 0),
            shape=(
                len(C66[:num_gauss_points_on_boundary]),
                len(C66[:num_gauss_points_on_boundary]),
            ),
            dtype=C66[:num_gauss_points_on_boundary].dtype,
        )
        .tocsr()
        .dot(grad_shape_func_b_y_times_det_J_b_time_weight_electrolyte)
    ).T @ (
        shape_func_b_electrolyte.multiply(normal_vector_z_electrolyte)
    )

    K33_mechanical_line_int = (
        (
            shape_func_times_det_J_time_weight_fixed_point
            @ sp.dia_array(
                (beta_Nitsche.reshape(1, -1), 0),
                shape=(len(beta_Nitsche), len(beta_Nitsche)),
                dtype=beta_Nitsche.dtype,
            ).tocsr()
        ).T
        @ shape_func_fixed_point
        - (
            sp.dia_array(
                (C55[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C55[:num_gauss_points_on_fixed_line]),
                    len(C55[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C55[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_x_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_x_electrolyte))
        - (
            sp.dia_array(
                (C66[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C66[:num_gauss_points_on_fixed_line]),
                    len(C66[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C66[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_y_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_y_electrolyte))
        - (
            sp.dia_array(
                (C33[:num_gauss_points_on_fixed_line].reshape(1, -1), 0),
                shape=(
                    len(C33[:num_gauss_points_on_fixed_line]),
                    len(C33[:num_gauss_points_on_fixed_line]),
                ),
                dtype=C33[:num_gauss_points_on_fixed_line].dtype,
            )
            .tocsr()
            .dot(grad_shape_func_z_times_det_J_time_weight_fixed_point)
        ).T
        @ (shape_func_fixed_point.multiply(normal_vector_z_electrolyte))
    )

    # legate_sparse requires same sparsity pattern for addition, so convert via dense
    if config.USE_NUMPY_EQUIVALENTS:
        K11_mechanical = csr_array(
            K11_mechanical_doamin_int.todense() + K11_mechanical_line_int.todense()
        )
        K12_mechanical = csr_array(
            K12_mechanical_doamin_int.todense() + K12_mechanical_line_int.todense()
        )
        K13_mechanical = csr_array(
            K13_mechanical_doamin_int.todense() + K13_mechanical_line_int.todense()
        )
        K21_mechanical = csr_array(
            K21_mechanical_doamin_int.todense() + K21_mechanical_line_int.todense()
        )
        K22_mechanical = csr_array(
            K22_mechanical_doamin_int.todense() + K22_mechanical_line_int.todense()
        )
        K23_mechanical = csr_array(
            K23_mechanical_doamin_int.todense() + K23_mechanical_line_int.todense()
        )
        K31_mechanical = csr_array(
            K31_mechanical_doamin_int.todense() + K31_mechanical_line_int.todense()
        )
        K32_mechanical = csr_array(
            K32_mechanical_doamin_int.todense() + K32_mechanical_line_int.todense()
        )
        K33_mechanical = csr_array(
            K33_mechanical_doamin_int.todense() + K33_mechanical_line_int.todense()
        )
    else:
        K11_mechanical = K11_mechanical_doamin_int + K11_mechanical_line_int
        K12_mechanical = K12_mechanical_doamin_int + K12_mechanical_line_int
        K13_mechanical = K13_mechanical_doamin_int + K13_mechanical_line_int
        K21_mechanical = K21_mechanical_doamin_int + K21_mechanical_line_int
        K22_mechanical = K22_mechanical_doamin_int + K22_mechanical_line_int
        K23_mechanical = K23_mechanical_doamin_int + K23_mechanical_line_int
        K31_mechanical = K31_mechanical_doamin_int + K31_mechanical_line_int
        K32_mechanical = K32_mechanical_doamin_int + K32_mechanical_line_int
        K33_mechanical = K33_mechanical_doamin_int + K33_mechanical_line_int

    if config.USE_NUMPY_EQUIVALENTS:
        K_mechanical = csr_array(
            np.block(
                [
                    [
                        K11_mechanical.todense(),
                        K12_mechanical.todense(),
                        K13_mechanical.todense(),
                    ],
                    [
                        K21_mechanical.todense(),
                        K22_mechanical.todense(),
                        K23_mechanical.todense(),
                    ],
                    [
                        K31_mechanical.todense(),
                        K32_mechanical.todense(),
                        K33_mechanical.todense(),
                    ],
                ]
            )
        )
    else:
        K_mechanical = bmat(
            [
                [K11_mechanical, K12_mechanical, K13_mechanical],
                [K21_mechanical, K22_mechanical, K23_mechanical],
                [K31_mechanical, K32_mechanical, K33_mechanical],
            ]
        )

    return K_mechanical


def mechanical_force_matrix_3d(
    x_G,
    C,
    epsilon_D1,
    epsilon_D2,
    epsilon_D3,
    epsilon_D4,
    epsilon_D5,
    epsilon_D6,
    grad_shape_func_x_times_det_J_time_weight,
    grad_shape_func_y_times_det_J_time_weight,
    grad_shape_func_z_times_det_J_time_weight,
):
    num_gauss_points = np.shape(x_G)[0]

    c_e_D1 = (
        C[0, 0] * epsilon_D1
        + C[0, 1] * epsilon_D2
        + C[0, 2] * epsilon_D3
        + C[0, 3] * epsilon_D4
        + C[0, 4] * epsilon_D5
        + C[0, 5] * epsilon_D6
    )
    c_e_D2 = (
        C[1, 0] * epsilon_D1
        + C[1, 1] * epsilon_D2
        + C[1, 2] * epsilon_D3
        + C[1, 3] * epsilon_D4
        + C[1, 4] * epsilon_D5
        + C[1, 5] * epsilon_D6
    )
    c_e_D3 = (
        C[2, 0] * epsilon_D1
        + C[2, 1] * epsilon_D2
        + C[2, 2] * epsilon_D3
        + C[2, 3] * epsilon_D4
        + C[2, 4] * epsilon_D5
        + C[2, 5] * epsilon_D6
    )
    c_e_D4 = (
        C[3, 0] * epsilon_D1
        + C[3, 1] * epsilon_D2
        + C[3, 2] * epsilon_D3
        + C[3, 3] * epsilon_D4
        + C[3, 4] * epsilon_D5
        + C[3, 5] * epsilon_D6
    )
    c_e_D5 = (
        C[4, 0] * epsilon_D1
        + C[4, 1] * epsilon_D2
        + C[4, 2] * epsilon_D3
        + C[4, 3] * epsilon_D4
        + C[4, 4] * epsilon_D5
        + C[4, 5] * epsilon_D6
    )
    c_e_D6 = (
        C[5, 0] * epsilon_D1
        + C[5, 1] * epsilon_D2
        + C[5, 2] * epsilon_D3
        + C[5, 3] * epsilon_D4
        + C[5, 4] * epsilon_D5
        + C[5, 5] * epsilon_D6
    )

    # assemble the force matrix for mechanical simulation
    f1_mechanical = (
        grad_shape_func_x_times_det_J_time_weight.T @ c_e_D1
        + grad_shape_func_y_times_det_J_time_weight.T @ c_e_D4
        + grad_shape_func_z_times_det_J_time_weight.T @ c_e_D5
    )
    f2_mechanical = (
        grad_shape_func_y_times_det_J_time_weight.T @ c_e_D2
        + grad_shape_func_x_times_det_J_time_weight.T @ c_e_D4
        + grad_shape_func_z_times_det_J_time_weight.T @ c_e_D6
    )
    f3_mechanical = (
        grad_shape_func_z_times_det_J_time_weight.T @ c_e_D3
        + grad_shape_func_x_times_det_J_time_weight.T @ c_e_D5
        + grad_shape_func_y_times_det_J_time_weight.T @ c_e_D6
    )
    f_mechanical = np.concatenate((f1_mechanical, f2_mechanical, f3_mechanical))
    # f_mechanical = f_mechanical[reorder_index, :]

    return f_mechanical
