# matrices.py
import numpy as np

# def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
#    """Example function with PEP 484 type annotations.
#
#    Args:
#        param1: The first parameter.
#        param2: The second parameter.
#
#    Returns:
#        The return value. True for success, False otherwise.
#
#    """

def npI() -> ndarray:
    """Returns the identity matrix in SU(2) algebra.

    Returns:
        The identity matrix as a numpy.array.

    """
    np_identity = np.array([[1, 0], [0, 1]])
    return np_identity

def npX() -> ndarray:
    """Returns the sigma_x matrix in SU(2) algebra.

    Returns:
        The sigma_x matrix as a numpy.array.

    """
    sigma_x = np.array([[0, 1], [1, 0]])
    return sigma_x

def npY() -> ndarray:
    """Returns the sigma_y matrix in SU(2) algebra.

    Returns:
        The sigma_y matrix as a numpy.array.

    """
    sigma_y = np.array([[0, -1j], [1j, 0]])
    return sigma_y

def npZ() -> ndarray:
    """Returns the sigma_z matrix in SU(2) algebra.

    Returns:
        The sigma_z matrix as a numpy.array.

    """
    sigma_z = np.array([[1, 0], [0, -1]])
    return sigma_z
