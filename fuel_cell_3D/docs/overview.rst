Overview
========

The **fuel_cell_3D** codebase implements a meshfree numerical method for simulating
3D solid oxide fuel cells (SOFCs). It uses the Reproducing Kernel Particle Method
(RKPM) to solve coupled diffusion and mechanical problems.

Code Structure
--------------

The codebase is organized into the following modules:

.. list-table:: Module Overview
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - ``main.py``
     - Main simulation driver; orchestrates the full simulation workflow
   * - ``config.py``
     - Configuration parameters (geometry, materials, numerical settings)
   * - ``common.py``
     - Backend abstraction (NumPy/SciPy or Legate/cuNumeric)
   * - ``read_image.py``
     - Reads voxelized microstructure from TIFF images
   * - ``get_nodes_gauss_points.py``
     - Generates node positions and Gauss quadrature points
   * - ``get_nodes_gauss_points_vectorized.py``
     - Vectorized implementations for Gauss point computations
   * - ``shape_function_in_domain.py``
     - Computes RKPM shape functions and their gradients
   * - ``shape_function_vectorized.py``
     - Vectorized shape function computations
   * - ``shape_grad_func_vectorized.py``
     - Vectorized shape function gradient computations
   * - ``define_diffusion_matrix_form.py``
     - Assembles diffusion stiffness matrix using Nitsche's method
   * - ``define_mechanical_stiffness_matrix.py``
     - Assembles mechanical stiffness matrix with elasticity tensor

Simulation Workflow
-------------------

The simulation proceeds through the following steps:

1. **Configuration**: Load parameters from ``config.py``
2. **Geometry Setup**: Read microstructure from TIFF image or generate synthetic geometry
3. **Node Generation**: Create nodes for each phase (electrolyte, electrode, pore)
4. **Gauss Point Generation**: Create integration points within each cell
5. **Shape Function Computation**: Compute RKPM shape functions and gradients
6. **Matrix Assembly**: Assemble diffusion and mechanical stiffness matrices
7. **Boundary Conditions**: Apply Dirichlet conditions via Nitsche's method
8. **Solve**: Solve the coupled system using sparse linear solvers
9. **Post-process**: Compute stress, strain, and electrochemical quantities

Backend Abstraction
-------------------

The code supports two computational backends:

1. **NumPy/SciPy**: Standard scientific Python stack (default)
2. **Legate/cuNumeric**: GPU-accelerated distributed computing

The backend is selected automatically in ``common.py``. The abstraction layer
provides consistent APIs:

.. code-block:: python

   from common import np, sparse, spsolve, csr_array

This allows the same code to run on CPUs or GPUs without modification.
