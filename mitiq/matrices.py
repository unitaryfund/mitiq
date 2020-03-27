# matrices.py
import numpy as np

def npI() -> np.ndarray:
    """Returns the identity matrix in SU(2) algebra.

    Returns:
        The identity matrix as a numpy.array.

    """
    np_identity = np.array([[1, 0], [0, 1]])
    return np_identity

def npX() -> np.ndarray:
    """Returns the sigma_x Pauli matrix in SU(2) algebra.

    Returns:
        The sigma_x matrix as a numpy.array.

    """
    sigma_x = np.array([[0, 1], [1, 0]])
    return sigma_x

def npY() -> np.ndarray:
    """Returns the sigma_y Pauli matrix in SU(2) algebra.

    Returns:
        The sigma_y matrix as a numpy.array.

    """
    sigma_y = np.array([[0, -1j], [1j, 0]])
    return sigma_y

def npZ() -> np.ndarray:
    """Returns the sigma_z Pauli matrix in SU(2) algebra.

    Returns:
        The sigma_z matrix as a numpy.array.

    """
    sigma_z = np.array([[1, 0], [0, -1]])
    return sigma_z
