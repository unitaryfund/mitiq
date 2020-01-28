
import numpy as np
from zne.utils import check_c, get_gammas




class Mitigator:
    """"Error mitigation class. 
    It can be used to process a quantum circuit in order to extrapolate an error-mitigated result.
    It requires a function circ_to_expval(circuit, stretch), defined by the user, which can execute a circuit 
    at a given noise scaling.
    """
        
    def __init__(self, circ_to_expval, order=0, c=None, method='richardson'):
        """Initializes the mitigator."""
        # consistency check of the arguments
        check_c(order, c)
        # initialize object variables
        self.circuit = None
        self.circ_to_expval = circ_to_expval
        self.order = order
        self.c = c
        self.method = method
        self.expvals = []
        self.result = None
    
    def load(self, circuit):
        """Loads the circuit into the mitigator object."""
        self.circuit = circuit
    
    def comp(self):
        """Compiles the circuit for error mitigation purposes."""
        # if the user is able to control the noise, do nothing.
        pass
    
    def run(self):
        """Executes the circuit for different noise levles"""
        _c = check_c(self.order, self.c)
        self.expvals = []
        for j, c_val in enumerate(_c):
            self.expvals.append(self.circ_to_expval(circuit=self.circuit, stretch=c_val))
        return self.expvals
    
    def extrapolate(self):
        """Extrapolates from the list self.expvals"""
        if self.method == 'richardson':
            _c = check_c(self.order, self.c)
            # get linear combination coefficients
            gammas = get_gammas(_c)
            # linear extraolation
            self.result = np.dot(gammas, self.expvals[0:self.order + 1])
        return self.result
    
    def __call__(self, circuit):
        """Evaluates the expectation value of the input circuit with error mitigation"""
        self.load(circuit)
        self.comp()
        self.run()
        self.extrapolate()
        return self.result

def mitigate(order=0, c=None, method='richardson'):
    """Decorator associated to the class Mitigator."""
    # more precisely, this is a wrap function which is necessary to pass parameters to the decorator
    # formally, the function below is the actual decorator
    def create_mitigator(fun):
        return Mitigator(fun, order=order, c=c, method=method)
    return create_mitigator  