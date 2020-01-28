"""Zero noise extrapolation sub-module."""

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