"""
Command line output of information on Mitiq and dependencies.
"""
__all__ = ['about']

import sys
import os
import platform
import numpy
import scipy
import mitiq
import inspect

def about():
    """
    About box for Mitiq. Gives version numbers for
    Mitiq, NumPy, SciPy, Cirq, PyQuil, Qiskit.
    """
    try:
        import cirq
        cirq_ver = cirq.__version__
    except:
        cirq_ver = 'None'
    try:
        import pyquil
        pyquil_ver = pyquil.__version__
    except:
        pyquil_ver = 'None'
    try:
        import qiskit
        qiskit_ver = qiskit.__version__
    except:
        qiskit_ver = 'None'
    print("")
    print("Mitiq: A Python toolkit for implementing ") 
    print("error mitigation on quantum computers.")
    print("========================================")
    print("Mitiq team – 2020 and later.")
    print("See https://github.com/unitaryfund/mitiq for details.")
    print("")
    print("Mitiq Version:      %s" % mitiq.__version__)
    print("Numpy Version:      %s" % numpy.__version__)
    print("Scipy Version:      %s" % scipy.__version__)
    print("Cirq Version:       %s" % cirq_ver)
    print("Pyquil Version:     %s" % pyquil_ver)
    print("Qiskit Version:     %s" % qiskit_ver)
    print("Python Version:     %d.%d.%d" % sys.version_info[0:3])
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    mitiq_install_path = os.path.dirname(inspect.getsourcefile(mitiq))
    print("Installation path:  %s" % mitiq_install_path)

if __name__ == "__main__":
    about()