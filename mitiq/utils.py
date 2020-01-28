# utils.py
import numpy as np

# used in Mitigator.extrapolate()
def get_gammas(c):
    """Returns the linear combination coefficients "gammas" for Richardson's extrapolation.
    The input is a list of the noise stretch factors.    
    """
    order = len(c) - 1
    np_c = np.asarray(c)
    A = np.zeros((order + 1, order + 1))
    for k in range(order + 1):
        A[k] = np_c ** k
    b = np.zeros(order + 1)
    b[0] = 1
    return np.linalg.solve(A, b)

# used in Mitigator.extrapolate()
def check_c(order, c):
    """Consistency check of the noise stretch vector. Returns c[:order + 1]. 
    If c is None, generates a default one."""
    if c == None:
        # generates a default list
        c = list(range(1, order + 2))
    if order > len(c) - 1:
        raise ValueError("Extrapolation order is too high compared to len(c) - 1.") 
    if c[0] != 1:
        raise ValueError("c[0] should be 1.") # Not sure if this is really a requirement.
    return c[:order + 1]
