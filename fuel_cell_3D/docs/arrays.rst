Array Shapes and Data Structures
=================================

This document describes the shapes and roles of the key arrays used in the
simulation. Understanding these structures is essential for development and
debugging.

Notation
--------

Throughout this document:

- ``n_nodes``: Number of nodes in a phase
- ``n_gauss``: Number of Gauss points in a domain
- ``n_gauss_b``: Number of Gauss points on a boundary
- ``n_cells``: Number of cells (voxels)
- ``n_nnz``: Number of non-zero entries in a sparse matrix
- ``dim``: Spatial dimension (3 for 3D)
- ``basis``: Polynomial basis size (4 for linear 3D: [1, x, y, z])

Node Arrays
-----------

.. list-table:: Node Coordinate Arrays
   :widths: 35 20 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``x_nodes_electrolyte``
     - ``(n_nodes, 3)``
     - Node coordinates [x, y, z] in electrolyte phase
   * - ``x_nodes_electrode``
     - ``(n_nodes, 3)``
     - Node coordinates [x, y, z] in electrode phase
   * - ``x_nodes_pore``
     - ``(n_nodes, 3)``
     - Node coordinates [x, y, z] in pore phase
   * - ``x_nodes_mechanical``
     - ``(n_nodes, 3)``
     - All nodes in the domain (union of all phases)
   * - ``nodes_grain_id_*``
     - ``(n_nodes,)``
     - Grain/phase ID for each node

Cell Connectivity Arrays
------------------------

.. list-table:: Cell Vertex Arrays (8 nodes per hexahedral cell)
   :widths: 40 20 40
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``cell_nodes_electrolyte_x``
     - ``(n_cells, 8)``
     - x-coordinates of 8 vertices for each electrolyte cell
   * - ``cell_nodes_electrolyte_y``
     - ``(n_cells, 8)``
     - y-coordinates of 8 vertices for each electrolyte cell
   * - ``cell_nodes_electrolyte_z``
     - ``(n_cells, 8)``
     - z-coordinates of 8 vertices for each electrolyte cell

Boundary cell arrays have 4 vertices per face:

.. list-table:: Boundary Cell Arrays (4 nodes per quadrilateral face)
   :widths: 45 20 35
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``cell_nodes_left_electrolyte_x``
     - ``(n_faces, 4)``
     - x-coords of boundary face vertices
   * - ``cell_nodes_interface_*_x/y/z``
     - ``(n_faces, 4)``
     - Interface cell vertex coordinates

Gauss Point Arrays
------------------

.. list-table:: Gauss Point Arrays
   :widths: 40 25 35
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``x_G_electrolyte``
     - ``(n_gauss, 3)``
     - Gauss point coordinates [x, y, z]
   * - ``det_J_time_weight_electrolyte``
     - ``(n_gauss,)``
     - det(J) × Gauss weight at each point
   * - ``x_G_b_electrolyte``
     - ``(n_gauss_b, 3)``
     - Boundary Gauss point coordinates
   * - ``Gauss_grain_id_*``
     - ``(n_gauss,)``
     - Phase ID at each Gauss point

For 3D hexahedral integration with 2×2×2 Gauss points:

- ``n_gauss = n_cells * 8``

For 2D boundary integration with 2x2 Gauss points:

- ``n_gauss_b = n_faces * 4``

Moment Matrices
---------------

.. list-table:: Moment Matrix Arrays
   :widths: 25 25 50
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``M``
     - ``(n_gauss, 4, 4)``
     - Moment matrix at each Gauss point (3D case)
   * - ``M_P_x``
     - ``(n_gauss, 4, 4)``
     - Partial derivative of M w.r.t. x
   * - ``M_P_y``
     - ``(n_gauss, 4, 4)``
     - Partial derivative of M w.r.t. y
   * - ``M_P_z``
     - ``(n_gauss, 4, 4)``
     - Partial derivative of M w.r.t. z

The moment matrix for 3D linear basis is 4×4:

.. math::

   \mathbf{M} = \begin{bmatrix}
   M_{00} & M_{01} & M_{02} & M_{03} \\
   M_{10} & M_{11} & M_{12} & M_{13} \\
   M_{20} & M_{21} & M_{22} & M_{23} \\
   M_{30} & M_{31} & M_{32} & M_{33}
   \end{bmatrix}

For 2D problems, the shape is ``(n_gauss, 3, 3)``.

Kernel Sparse Arrays
--------------------

The kernel values φ are stored in COO format for sparse matrix construction:

.. list-table:: Kernel Value Arrays (Sparse)
   :widths: 35 20 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``phi_nonzero_index_row``
     - ``(n_nnz,)``
     - Gauss point indices (row indices)
   * - ``phi_nonzero_index_column``
     - ``(n_nnz,)``
     - Node indices (column indices)
   * - ``phi_nonzerovalue_data``
     - ``(n_nnz,)``
     - Kernel values φ(z)
   * - ``phi_P_x_nonzerovalue_data``
     - ``(n_nnz,)``
     - ∂φ/∂x values
   * - ``phi_P_y_nonzerovalue_data``
     - ``(n_nnz,)``
     - ∂φ/∂y values
   * - ``phi_P_z_nonzerovalue_data``
     - ``(n_nnz,)``
     - ∂φ/∂z values

These arrays are used to construct sparse matrices:

.. code-block:: python

   phi_sparse = csr_array((phi_data, (row_indices, col_indices)), 
                          shape=(n_gauss, n_nodes))

Shape Function Sparse Matrices
------------------------------

.. list-table:: Shape Function Matrices (CSR Format)
   :widths: 45 25 30
   :header-rows: 1

   * - Matrix Name
     - Shape
     - Description
   * - ``shape_func``
     - ``(n_gauss, n_nodes)``
     - Shape functions Ψ_I(x_g)
   * - ``shape_func_times_det_J_weight``
     - ``(n_gauss, n_nodes)``
     - Ψ × det(J) × w
   * - ``grad_shape_func_x``
     - ``(n_gauss, n_nodes)``
     - ∂Ψ/∂x
   * - ``grad_shape_func_y``
     - ``(n_gauss, n_nodes)``
     - ∂Ψ/∂y
   * - ``grad_shape_func_z``
     - ``(n_gauss, n_nodes)``
     - ∂Ψ/∂z
   * - ``grad_shape_func_*_times_det_J_weight``
     - ``(n_gauss, n_nodes)``
     - ∂Ψ/∂* × det(J) × w

**Typical sparsity:** 

Each Gauss point interacts with ~20-50 nodes (depending on support size),
so the fill ratio is approximately ``30/n_nodes``.

Stiffness Matrices
------------------

.. list-table:: System Matrices (CSR Format)
   :widths: 30 25 45
   :header-rows: 1

   * - Matrix Name
     - Shape
     - Description
   * - ``K_electrolyte``
     - ``(n_nodes, n_nodes)``
     - Diffusion stiffness for electrolyte
   * - ``K_electrode``
     - ``(n_nodes, n_nodes)``
     - Diffusion stiffness for electrode
   * - ``K_pore``
     - ``(n_nodes, n_nodes)``
     - Diffusion stiffness for pore
   * - ``K_mechanical``
     - ``(3*n_nodes, 3*n_nodes)``
     - Mechanical stiffness (3 DOF per node)

The mechanical stiffness is assembled as a 3×3 block matrix:

.. math::

   \mathbf{K}_{mech} = \begin{bmatrix}
   K_{uu} & K_{uv} & K_{uw} \\
   K_{vu} & K_{vv} & K_{vw} \\
   K_{wu} & K_{wv} & K_{ww}
   \end{bmatrix}

where each block is ``(n_nodes, n_nodes)``.

Elasticity Tensor
-----------------

.. list-table:: Elasticity Tensor Arrays
   :widths: 25 30 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``C``
     - ``(6, 6, n_gauss, 1)``
     - Voigt elasticity tensor at Gauss points
   * - ``C11``, ``C12``, etc.
     - ``(n_gauss, 1)``
     - Individual tensor components

Voigt notation mapping (3D):

- Index 1: σ_xx, ε_xx
- Index 2: σ_yy, ε_yy
- Index 3: σ_zz, ε_zz
- Index 4: σ_yz, 2ε_yz
- Index 5: σ_xz, 2ε_xz
- Index 6: σ_xy, 2ε_xy

Solution Vectors
----------------

.. list-table:: Solution Arrays
   :widths: 30 25 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``phi_electrolyte``
     - ``(n_nodes,)``
     - Ionic potential in electrolyte
   * - ``phi_electrode``
     - ``(n_nodes,)``
     - Electronic potential in electrode
   * - ``c_pore``
     - ``(n_nodes,)``
     - Gas concentration in pores
   * - ``u_mechanical``
     - ``(3*n_nodes,)``
     - Displacement [u_x, u_y, u_z] stacked

Force Vectors
-------------

.. list-table:: Force/RHS Vectors
   :widths: 30 25 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``f_electrolyte``
     - ``(n_nodes,)``
     - RHS for electrolyte diffusion
   * - ``f_mechanical``
     - ``(3*n_nodes,)``
     - RHS for mechanical problem
   * - ``point_or_line_source``
     - ``(n_source,)``
     - Electrochemical source values
   * - ``interface_source``
     - ``(n_interface,)``
     - Interface reaction source

Boundary Arrays
---------------

.. list-table:: Boundary Condition Arrays
   :widths: 35 20 45
   :header-rows: 1

   * - Array Name
     - Shape
     - Description
   * - ``g_dirichlet``
     - ``(n_gauss_b,)``
     - Prescribed boundary values
   * - ``beta_Nitsche``
     - ``(n_gauss_b,)``
     - Nitsche penalty parameter
   * - ``normal_vector_x``
     - ``(n_gauss_b,)``
     - x-component of outward normal
   * - ``normal_vector_y``
     - ``(n_gauss_b,)``
     - y-component of outward normal
   * - ``normal_vector_z``
     - ``(n_gauss_b,)``
     - z-component of outward normal

Gauss Integration Reference
---------------------------

**3D Cube (8 Gauss points):**

The Gauss points are located at ξ = ±1/√3 in each direction:

.. code-block:: text

   X_G_CUBE coordinates: (+-0.577, +-0.577, +-0.577)
   Shape: (8, 3)
   WEIGHT_G_CUBE = [1, 1, 1, 1, 1, 1, 1, 1]  # Shape: (8,)

**2D Rectangle (4 Gauss points):**

.. code-block:: text

   X_G_REC coordinates: (+-0.577, +-0.577)
   Shape: (4, 2)
   WEIGHT_G_REC = [1, 1, 1, 1]  # Shape: (4,)
