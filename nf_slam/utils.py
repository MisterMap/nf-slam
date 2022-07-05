import flax.core.frozen_dict
import jaxlib.xla_extension


def size(a):
    if type(a) == jaxlib.xla_extension.DeviceArray:
        return a.nbytes
    elif type(a) == flax.core.frozen_dict.FrozenDict:
        summa = 0
        for b in a.values():
            summa += size(b)
        return summa
    else:
        summa = 0
        for b in a.__dict__.values():
            summa += size(b)
        return summa
