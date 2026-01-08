Implementation Details
======================

This document describes the implementation details of each module, including
the computational flow and key algorithms.

Configuration (config.py)
-------------------------

The configuration module defines all simulation parameters:

**Geometry Settings:**

- ``SINGLE_GRAIN``: Boolean to use single grain vs. image-based geometry
- ``DIMENSION``: Problem dimension (3 for 3D)
- ``X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX``: Domain boundaries (meters)

**Material Properties:**

- ``DIFFUSION_ELECTROLYTE``, ``DIFFUSION_ELECTRODE``, ``DIFFUSION_PORE``: Diffusion coefficients
- ``E_ELECTROLYTE``, ``E_ELECTRODE``: Young's modulus (Pa)
- ``NU_ELECTROLYTE``, ``NU_ELECTRODE``: Poisson's ratio
- ``LAMBDA_MECHANICAL_*``, ``MU_*``: Computed Lamé constants

**Electrochemical Parameters:**

- ``FARADAY_CONSTANT``: 96,485 C/mol
- ``GAS_CONSTANT``: 8.314 J/mol·K
- ``I_0``: Exchange current density
- ``TEMPERATURE``: Operating temperature (K)

Backend Abstraction (common.py)
-------------------------------

The ``common.py`` module provides a unified interface for NumPy/SciPy or
Legate/cuNumeric backends:

.. code-block:: python

   # Attempts to import Legate, falls back to NumPy/SciPy
   try:
       import cupynumeric as np
       import legate_sparse as sparse
   except ImportError:
       import numpy as np
       import scipy.sparse as sparse

**Exported Symbols:**

- ``np``: NumPy or cuNumeric array operations
- ``sparse``, ``sp``: Sparse matrix module
- ``csr_array``, ``dia_array``: Sparse matrix constructors
- ``spsolve``: Sparse linear solver
- ``block_diag``, ``vstack``, ``bmat``: Sparse matrix utilities
- ``time()``: High-resolution timer function

Image Reading (read_image.py)
-----------------------------

The ``read_in_image()`` function reads voxelized microstructure data:

**Input:**

- TIFF file with phase labels (0: pore, 1: electrolyte, 2: electrode)

**Output:**

- ``img_``: 3D NumPy array of phase IDs
- ``unic_grain_id``: List of unique phase IDs
- ``num_pixels_xyz``: [nx, ny, nz] voxel counts

.. code-block:: python

   def read_in_image(file_name, studied_physics, dimension):
       img_ = tifffile.imread(file_name)
       # Extract unique grain IDs and dimensions
       return img_, unic_grain_id, num_pixels_xyz

Node and Gauss Point Generation
-------------------------------

get_nodes_gauss_points.py
^^^^^^^^^^^^^^^^^^^^^^^^^

**Key Function: ``get_x_nodes_fuel_cell_3d_toy_image()``**

Generates nodes and cell connectivity from voxelized geometry:

1. Creates nodes at voxel corners for each phase
2. Identifies boundary cells on domain faces
3. Detects interface cells between phases
4. Returns cell vertex coordinates for integration

**Output arrays:**

- ``x_nodes_*``: Node coordinates for each phase
- ``cell_nodes_*_x/y/z``: Cell vertex coordinates (n_cells, 8)
- ``nodes_id_*``: Node IDs on boundaries

get_nodes_gauss_points_vectorized.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function: ``x_G_and_def_J_time_weight_3d_fuelcell_domain_vectorized()``**

Computes Gauss point coordinates and Jacobian-weighted measures for 3D hexahedral cells:

1. Evaluates 8-node trilinear shape functions at Gauss points
2. Computes Jacobian matrix from shape function derivatives
3. Returns physical coordinates and det(J) × weight

.. code-block:: python

   # Shape functions for 8-node hexahedron
   N[:, 0] = (1 + xi) * (1 - eta) * (1 - zeta) / 8.0
   N[:, 1] = (1 + xi) * (1 + eta) * (1 - zeta) / 8.0
   # ... (8 shape functions total)

   # Jacobian determinant
   det_J = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31)

Shape Function Computation
--------------------------

shape_function_in_domain.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function: ``compute_phi_M()``**

Computes RKPM kernel values and moment matrices:

1. For each Gauss point, loops over nodes within support radius
2. Computes normalized distance :math:`z = \|x - x_I\| / a`
3. Evaluates cubic B-spline kernel :math:`\phi(z)`
4. Accumulates moment matrix :math:`M` and derivatives

**Kernel Evaluation:**

.. code-block:: python

   if z >= 0 and z < 0.5:
       phi = 2.0/3 - 4*z**2 + 4*z**3
       phi_P_z = -8.0*z + 12.0*z**2
   elif z >= 0.5 and z <= 1:
       phi = 4.0/3 - 4*z + 4*z**2 - 4.0/3*z**3
       phi_P_z = -4 + 8*z - 4*z**2

**Function: ``shape_grad_shape_func()``**

Computes shape function values and gradients:

.. math::

   \Psi_I = \mathbf{H}^T_0 \mathbf{M}^{-1} \mathbf{H} \phi

   \nabla \Psi_I = \mathbf{H}^T_0 \mathbf{M}^{-1} \mathbf{H} \nabla\phi 
                 + \mathbf{H}^T_0 \nabla\mathbf{M}^{-1} \mathbf{H} \phi
                 + \mathbf{H}^T_0 \mathbf{M}^{-1} \nabla\mathbf{H} \phi

shape_function_vectorized.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function: ``compute_phi_M_standard_sparse()``**

Optimized vectorized implementation using advanced NumPy indexing:

1. Computes all Gauss-node distances at once using broadcasting
2. Uses ``np.where()`` to find valid pairs (z ≤ 1)
3. Accumulates M matrices using ``np.add.at()`` for efficiency

.. code-block:: python

   # Vectorized distance computation
   diff = x_G[:, np.newaxis, :] - x_nodes[np.newaxis, :, :]  # (n_gauss, n_nodes, 3)
   distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_gauss, n_nodes)
   z = distances / a[np.newaxis, :]  # Normalized distances

   # Find valid pairs
   valid_mask = (z >= 0) & (z <= 1.0)
   gauss_indices, node_indices = np.where(valid_mask)

Diffusion Matrix Assembly
-------------------------

define_diffusion_matrix_form.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function: ``diffusion_matrix_fuel_cell()``**

Assembles the diffusion stiffness matrix using Nitsche's method:

**Stiffness Matrix Components:**

.. code-block:: python

   # K1: Domain integral
   K1 = (grad_shape_x_weighted.T * D) @ grad_shape_x + (grad_shape_y_weighted.T * D) @ grad_shape_y

   # K2: Flux consistency term
   K2 = -(grad_shape_b_x_weighted * n_x + grad_shape_b_y_weighted * n_y).T @ shape_b

   # K3: Nitsche penalty term
   K3 = (shape_b * beta).T @ shape_b_weighted

   K = K1 + K2 + K3

**Force Vector Components:**

.. code-block:: python

   # f1: Boundary value penalty
   f1 = (shape_b_weighted * beta).T @ g_dirichlet

   # f2: Flux contribution
   f2 = -(grad_shape_b_weighted * n).T @ g_dirichlet

   # f3: Point/body source
   f3 = shape_func_source.T @ point_source

   f = f1 + f2 + f3

Mechanical Stiffness Assembly
-----------------------------

define_mechanical_stiffness_matrix.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function: ``mechanical_C_tensor_3d()``**

Constructs the 6×6 elasticity tensor in Voigt notation:

1. Computes Lamé constants from E and ν
2. Applies rotation for anisotropic materials
3. Applies damage degradation: :math:`C_{eff} = (1-D) C`

**Function: ``mechanical_stiffness_matrix_3d_fuel_cell()``**

Assembles the 3×3 block mechanical stiffness matrix:

.. math::

   \mathbf{K}_{mech} = \begin{bmatrix}
   K_{11} & K_{12} & K_{13} \\
   K_{21} & K_{22} & K_{23} \\
   K_{31} & K_{32} & K_{33}
   \end{bmatrix}

Each block involves combinations of stress components:

.. code-block:: python

   # K11: Coupling of u_x with u_x
   K11 = (C11 * grad_x_weighted).T @ grad_x 
       + (C44 * grad_y_weighted).T @ grad_y 
       + (C55 * grad_z_weighted).T @ grad_z

   # K12: Coupling of u_x with u_y
   K12 = (C12 * grad_x_weighted).T @ grad_y 
       + (C44 * grad_y_weighted).T @ grad_x

Main Simulation Flow (main.py)
------------------------------

The main script orchestrates the full simulation:

1. **Setup Phase** (lines 1-350):
   - Load configuration
   - Read microstructure image
   - Generate nodes for each phase
   - Create cell connectivity

2. **Gauss Point Generation** (lines 360-550):
   - Compute domain Gauss points and Jacobians
   - Compute boundary Gauss points
   - Compute interface Gauss points

3. **Shape Function Computation** (lines 550-1500):
   - Compute kernel values φ for all Gauss-node pairs
   - Assemble moment matrices M, M_P_x, M_P_y, M_P_z
   - Compute shape functions and gradients
   - Build sparse shape function matrices

4. **Matrix Assembly** (lines 1500-2500):
   - Assemble diffusion matrices for each phase
   - Apply boundary conditions
   - Assemble mechanical stiffness matrix
   - Build coupled system

5. **Solution** (lines 2500-3500):
   - Solve diffusion equations (electrolyte, electrode, pore)
   - Compute electrochemical source terms
   - Iterate for coupled solution
   - Solve mechanical problem

6. **Post-processing** (lines 3500-4400):
   - Compute stress and strain fields
   - Generate visualization plots
   - Output results
