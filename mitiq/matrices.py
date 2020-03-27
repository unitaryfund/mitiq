# matrices.py
import numpy as np

npI = np.array([[1, 0], [0, 1]])
"""Defines the identity matrix in SU(2) algebra as a (2,2) `np.array`."""

npX = np.array([[0, 1], [1, 0]])
"""Defines the sigma_x Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npY = np.array([[0, -1j], [1j, 0]])
"""Defines the sigma_y Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""

npZ = np.array([[1, 0], [0, -1]])
"""Defines the sigma_z Pauli matrix in SU(2) algebra as a (2,2) `np.array`."""
