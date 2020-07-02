import numpy as np
import pyarrow as pa
import vaex
from ..expression import _binary_ops, reversable

class NumpyDispatch:
    def __init__(self, arrow_array):
        self.arrow_array = arrow_array
        self.numpy_array = vaex.array_types.to_numpy(self.arrow_array)


for op in _binary_ops:
    def closure(op=op):
        def operator(a, b):
            return NumpyDispatch(pa.array(op['op'](a.numpy_array, b.numpy_array)))
        return operator
    method_name = '__%s__' % op['name']
    setattr(NumpyDispatch, method_name, closure())
    # to support e.g. (1 + ...)
    if op['name'] in reversable:
        def closure(op=op):
            def operator(b, a):
                return NumpyDispatch(pa.array(op['op'](a, b.numpy_array)))
            return operator
        method_name = '__r%s__' % op['name']
        setattr(NumpyDispatch, method_name, closure())
