Physics and Mathematical Formulation
====================================

This document describes the physical models and mathematical formulations
implemented in the fuel_cell_3D code.

Reproducing Kernel Particle Method (RKPM)
-----------------------------------------

RKPM is a meshfree method that constructs shape functions using a reproducing
kernel approximation. The displacement field is approximated as:

.. math::

   u^h(\mathbf{x}) = \sum_{I=1}^{n_p} \Psi_I(\mathbf{x}) \, d_I

where :math:`\Psi_I(\mathbf{x})` is the shape function for node :math:`I` and
:math:`d_I` are the nodal coefficients.

Shape Function Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RKPM shape function is defined as:

.. math::

   \Psi_I(\mathbf{x}) = \mathbf{H}^T(\mathbf{0}) \, \mathbf{M}^{-1}(\mathbf{x}) \, 
   \mathbf{H}(\mathbf{x} - \mathbf{x}_I) \, \phi_a\left(\frac{\|\mathbf{x} - \mathbf{x}_I\|}{a_I}\right)

where:

- :math:`\mathbf{H}(\mathbf{x})` is the polynomial basis vector
- :math:`\mathbf{M}(\mathbf{x})` is the moment matrix
- :math:`\phi_a(z)` is the kernel function
- :math:`a_I` is the support size for node :math:`I`

For 3D problems, the linear polynomial basis is:

.. math::

   \mathbf{H}(\mathbf{x}) = \begin{bmatrix} 1 \\ x/h \\ y/h \\ z/h \end{bmatrix}

where :math:`h = 10^{-6}` m is a scaling factor for numerical stability.

The moment matrix is computed as:

.. math::

   \mathbf{M}(\mathbf{x}) = \sum_{I=1}^{n_p} \mathbf{H}(\mathbf{x} - \mathbf{x}_I) \, 
   \mathbf{H}^T(\mathbf{x} - \mathbf{x}_I) \, \phi_a(z_{xI})

Kernel Function
^^^^^^^^^^^^^^^

The cubic B-spline kernel is used:

.. math::

   \phi(z) = \begin{cases}
   \frac{2}{3} - 4z^2 + 4z^3 & 0 \leq z < 0.5 \\
   \frac{4}{3} - 4z + 4z^2 - \frac{4}{3}z^3 & 0.5 \leq z \leq 1 \\
   0 & z > 1
   \end{cases}

where :math:`z = \|\mathbf{x} - \mathbf{x}_I\| / a_I` is the normalized distance.

Diffusion Equation
------------------

The steady-state diffusion equation governs species transport:

.. math::

   \nabla \cdot (D \nabla c) + f = 0 \quad \text{in } \Omega

where:

- :math:`c` is the concentration (or potential)
- :math:`D` is the diffusion coefficient
- :math:`f` is the source term

Weak Form
^^^^^^^^^

The weak form is:

.. math::

   \int_\Omega D \nabla w \cdot \nabla c \, d\Omega - \int_{\Gamma_N} w q \, d\Gamma 
   - \int_\Omega w f \, d\Omega = 0

where :math:`w` is the test function and :math:`q` is the prescribed flux.

Nitsche's Method for Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dirichlet boundary conditions are enforced weakly using Nitsche's method:

.. math::

   \int_{\Gamma_D} \beta (c - g) w \, d\Gamma - \int_{\Gamma_D} D \nabla c \cdot \mathbf{n} \, w \, d\Gamma = 0

where:

- :math:`g` is the prescribed boundary value
- :math:`\beta` is the Nitsche penalty parameter
- :math:`\mathbf{n}` is the outward normal vector

The discretized stiffness matrix has three components:

.. math::

   \mathbf{K} = \mathbf{K}_1 + \mathbf{K}_2 + \mathbf{K}_3

where:

- :math:`\mathbf{K}_1`: Domain diffusion integral
- :math:`\mathbf{K}_2`: Consistency term (flux boundary)
- :math:`\mathbf{K}_3`: Nitsche penalty term

Butler-Volmer Electrochemistry
------------------------------

The electrochemical reaction rate at the electrode-electrolyte interface
follows the Butler-Volmer equation:

.. math::

   i = i_0 \left[ \exp\left(\frac{\alpha_a F \eta}{RT}\right) 
   - \exp\left(-\frac{\alpha_c F \eta}{RT}\right) \right]

where:

- :math:`i` is the current density
- :math:`i_0` is the exchange current density
- :math:`\eta` is the overpotential
- :math:`F` is Faraday's constant (96,485 C/mol)
- :math:`R` is the gas constant (8.314 J/mol·K)
- :math:`T` is the temperature
- :math:`\alpha_a, \alpha_c` are the anodic and cathodic transfer coefficients

For small overpotentials, the linearized form is:

.. math::

   i \approx i_0 \frac{F \eta}{RT}

Linear Elasticity
-----------------

The mechanical equilibrium equation is:

.. math::

   \nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = 0 \quad \text{in } \Omega

where :math:`\boldsymbol{\sigma}` is the Cauchy stress tensor and :math:`\mathbf{b}` is the body force.

Constitutive Relation
^^^^^^^^^^^^^^^^^^^^^

The stress-strain relationship for linear elasticity:

.. math::

   \boldsymbol{\sigma} = \mathbf{C} : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^*)

where:

- :math:`\mathbf{C}` is the fourth-order elasticity tensor
- :math:`\boldsymbol{\varepsilon}` is the total strain
- :math:`\boldsymbol{\varepsilon}^*` is the chemical expansion strain

Elasticity Tensor
^^^^^^^^^^^^^^^^^

For isotropic materials, the elasticity tensor in Voigt notation is:

.. math::

   \mathbf{C} = \begin{bmatrix}
   \lambda + 2\mu & \lambda & \lambda & 0 & 0 & 0 \\
   \lambda & \lambda + 2\mu & \lambda & 0 & 0 & 0 \\
   \lambda & \lambda & \lambda + 2\mu & 0 & 0 & 0 \\
   0 & 0 & 0 & \mu & 0 & 0 \\
   0 & 0 & 0 & 0 & \mu & 0 \\
   0 & 0 & 0 & 0 & 0 & \mu
   \end{bmatrix}

where the Lamé constants are:

.. math::

   \lambda = \frac{E \nu}{(1+\nu)(1-2\nu)}, \quad \mu = \frac{E}{2(1+\nu)}

with :math:`E` being Young's modulus and :math:`\nu` Poisson's ratio.

Chemical Expansion Strain
^^^^^^^^^^^^^^^^^^^^^^^^^

The chemical expansion strain due to concentration changes:

.. math::

   \varepsilon^*_{ij} = \frac{\beta}{3} \Delta c \, \delta_{ij}

where :math:`\beta` is the chemical expansion coefficient (m³/mol) and
:math:`\Delta c` is the concentration change.

Gauss Quadrature
----------------

Domain integrals are evaluated using Gauss quadrature:

.. math::

   \int_\Omega f(\mathbf{x}) \, d\Omega \approx \sum_{g=1}^{n_g} f(\mathbf{x}_g) \, |J_g| \, w_g

where:

- :math:`\mathbf{x}_g` are the Gauss point coordinates
- :math:`|J_g|` is the Jacobian determinant
- :math:`w_g` are the Gauss weights

For 3D hexahedral cells, 2×2×2 Gauss integration is used with points at:

.. math::

   \xi_i = \pm \frac{1}{\sqrt{3}}, \quad w_i = 1

Damage Model
------------

When enabled, a damage variable :math:`D` modifies the stiffness:

.. math::

   \mathbf{C}_{eff} = (1 - D) \mathbf{C}

The damage variable evolves based on equivalent strain:

.. math::

   D = \begin{cases}
   0 & \varepsilon_{eq} < \kappa_i \\
   1 - \frac{\kappa_i}{\varepsilon_{eq}} \exp\left(-\frac{\varepsilon_{eq} - \kappa_i}{\kappa_f - \kappa_i}\right) & \varepsilon_{eq} \geq \kappa_i
   \end{cases}

where :math:`\kappa_i` and :math:`\kappa_f` are the initial and final damage thresholds.
