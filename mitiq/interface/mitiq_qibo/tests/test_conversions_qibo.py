from mitiq.interface.mitiq_qibo.conversions import from_qibo, to_qibo
from mitiq.zne.scaling import fold_gates_from_left
from qibo.models import Circuit as qibo_Circuit
from qibo import gates


c = qibo_Circuit(2)
c.add(gates.X(0))
mitiq_c = from_qibo(c)
print(mitiq_c)
#folded_mitiq = fold_gates_from_left(mitiq_c, scale_factor=4.)
#print(folded_mitiq)

#assert to_qibo(mitiq_c) == c

