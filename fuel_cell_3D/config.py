"""
Configuration file for fuel cell 3D simulation.
Contains all configuration parameters including domain dimensions,
physical constants, material properties, and simulation settings.
"""

# Import NumPy from common.py
from common import np

# ==============================================================================
# GEOMETRY AND ANALYSIS SETTINGS
# ==============================================================================

# Geometry settings
SINGLE_GRAIN = False  # True: single grain, False: read from an image
DIMENSION = 3  # 3D or 2D simulation

# Analysis type
IM_RKPM = False  # If it is interfacial modified RKPM (only available for battery)
STUDIED_PHYSICS = "fuel cell"  # "fuel cell" or "battery"
DAMAGE_MODEL = "OFF"  # "ON" or "OFF"

# Point source configuration
DELTA_POINT_SOURCE = (
    True  # If point source is delta function. If distributed, set to False
)

# ==============================================================================
# OUTPUT CONTROL FLAGS
# ==============================================================================

# Control whether to enable plotting (matplotlib)
ENABLE_PLOTTING = True  # Set to True to show plots, False to disable all plotting

# Control whether to enable file I/O operations (saving data to files)
ENABLE_FILE_IO = True  # Set to True to save output files, False to disable file I/O

# ==============================================================================
# NUMERICAL METHODS
# ==============================================================================

# Differential and integral methods
DIFFERENTIAL_METHOD = "direct"  # "implicit" or "direct"
# Note: if IM_RKPM=True, differential method must be set to "direct"
INTEGRAL_METHOD = "gauss"  # Integration method

# ==============================================================================
# DOMAIN DIMENSIONS (in meters)
# ==============================================================================

# 3D domain boundaries
X_MIN = 0.0
X_MAX = 10e-6
Y_MIN = 0.0
Y_MAX = 20e-6
Z_MIN = 0.0
Z_MAX = 10e-6

# ==============================================================================
# MATERIAL ORIENTATION
# ==============================================================================

# Grain angle
ANGLE = 0.0  # Grain orientation angle (in radians)

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

FARADAY_CONSTANT = 9.6485e4  # Faraday constant (C/mol)
GAS_CONSTANT = 8.3145  # Universal gas constant (J/mol·K)

# ==============================================================================
# MATERIAL PROPERTIES - DIFFUSION
# ==============================================================================

# Diffusion coefficients
DIFFUSION_ELECTROLYTE = 0.035
DIFFUSION_ELECTRODE = 1.0e-9
DIFFUSION_PORE = 1.0e-7

# Gas permeability
K_GAS = 1.0e-4

# ==============================================================================
# ELECTROCHEMICAL PARAMETERS
# ==============================================================================

# Exchange current densities
I_0 = 1.0e-1  # Exchange current density
I_0_SOLID = 1.0e1  # Solid phase exchange current density

# Temperature
TEMPERATURE = 1273.2  # Operating temperature (K)

# Potentials
E_0 = 1.0  # Standard potential (V)
V_APP = 1.5  # Applied voltage (V)

# Boundary concentrations
C_BOUNDARY = 1000.0  # Concentration at boundary
C_BOUNDARY_PORE = 9.572  # Pore concentration at boundary

# ==============================================================================
# MECHANICAL PROPERTIES
# ==============================================================================

# Electrolyte mechanical properties
E_ELECTROLYTE = 132.69e9  # Young's modulus (Pa)
NU_ELECTROLYTE = 0.33  # Poisson's ratio
LAMBDA_MECHANICAL_ELECTROLYTE = (
    E_ELECTROLYTE * NU_ELECTROLYTE / (1 + NU_ELECTROLYTE) / (1 - 2 * NU_ELECTROLYTE)
)  # First Lamé constant
MU_ELECTROLYTE = (
    E_ELECTROLYTE / 2 / (1 + NU_ELECTROLYTE)
)  # Second Lamé constant (shear modulus)

# Electrode mechanical properties
E_ELECTRODE = 130.0e9  # Young's modulus (Pa)
NU_ELECTRODE = 0.33  # Poisson's ratio
LAMBDA_MECHANICAL_ELECTRODE = (
    E_ELECTRODE * NU_ELECTRODE / (1 + NU_ELECTRODE) / (1 - 2 * NU_ELECTRODE)
)  # First Lamé constant
MU_ELECTRODE = (
    E_ELECTRODE / 2 / (1 + NU_ELECTRODE)
)  # Second Lamé constant (shear modulus)

# Expansion coefficient
BETA_FUELCELL_EXPANSION_COEFFICIENT = 4.0e-6  # Thermal expansion coefficient (m³/mol)

# ==============================================================================
# DAMAGE MODEL PARAMETERS
# ==============================================================================

# Damage parameters (used when DAMAGE_MODEL = "ON")
K_I = 0.0125  # Initial damage parameter
K_F = 0.015  # Final damage parameter

# ==============================================================================
# IMAGE FILE CONFIGURATION
# ==============================================================================

# Image file for geometry (used when SINGLE_GRAIN = False)
IMAGE_FILE_NAME = "M_3d_3phases_2K.tif"  # 2K Voxels
# IMAGE_FILE_NAME = "M_3d_3phases_16K.tif"  # 16K Voxels

# ==============================================================================
# GAUSS INTEGRATION POINTS AND WEIGHTS
# ==============================================================================

# 3D cube Gauss points (8 points)
X_G_CUBE = np.array(
    [
        [-(3**0.5) / 3, -(3**0.5) / 3, -(3**0.5) / 3],
        [3**0.5 / 3, -(3**0.5) / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, 3**0.5 / 3, -(3**0.5) / 3],
        [3**0.5 / 3, 3**0.5 / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, -(3**0.5) / 3, 3**0.5 / 3],
        [3**0.5 / 3, -(3**0.5) / 3, 3**0.5 / 3],
        [-(3**0.5) / 3, 3**0.5 / 3, 3**0.5 / 3],
        [3**0.5 / 3, 3**0.5 / 3, 3**0.5 / 3],
    ]
)  # Coordinates of 3D Gauss points in neutral coordinate system

WEIGHT_G_CUBE = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)  # Weights for 3D cube

# 2D rectangle Gauss points (4 points)
X_G_REC = np.array(
    [
        [-(3**0.5) / 3, -(3**0.5) / 3],
        [-(3**0.5) / 3, 3**0.5 / 3],
        [3**0.5 / 3, -(3**0.5) / 3],
        [3**0.5 / 3, 3**0.5 / 3],
    ]
)  # Coordinates of 2D Gauss points for rectangle

WEIGHT_G_REC = np.array([1.0, 1.0, 1.0, 1.0])  # Weights for 2D rectangle

# 2D triangle Gauss points (3 points)
X_G_TRI = np.array(
    [
        [1.0 / 6.0, 2.0 / 3.0],
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
    ]
)  # Coordinates of 2D Gauss points for triangle

WEIGHT_G_TRI = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])  # Weights for 2D triangle

# 1D line Gauss points (4 points)
X_G_LINE = np.array(
    [
        -((3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
        -((3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5),
        (3.0 / 7.0 - 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
        (3.0 / 7.0 + 2.0 / 7.0 * (1.2) ** 0.5) ** 0.5,
    ]
)  # Coordinates of 1D Gauss points

WEIGHT_G_LINE = np.array(
    [
        0.5 - 30**0.5 / 36,
        0.5 + 30**0.5 / 36,
        0.5 + 30**0.5 / 36,
        0.5 - 30**0.5 / 36,
    ]
)  # Weights for 1D line

# ======================
# PERFORMANCE RELATED
# ======================

# Allow using NumPy APIs for APIs that are not supported by legate-sparse
# e.g., vstack, block_diag, block_array etc
# NOTE: THIS HAS TO BE TRUE if we want to use legate
USE_NUMPY_EQUIVALENTS = True
