from enum import Enum
import numpy as np

class DTYPE(Enum):
    F32 = (1, np.float32)
    U8 = (2, np.uint8)
    I8 = (3, np.int8)
    U16 = (4, np.uint16)
    I16 = (5, np.int16)
    I32 = (6, np.int32)
    I64 = (7, np.int64)
    String = (8, np.string_)
    Bool = (9, np.bool_)
    FP16 = (10, np.float16)
    F64 = (11, np.float64)
    U32 = (12, np.uint32)
    U64 = (13, np.uint64)
    Complex64 = (14, np.complex64)
    Complex128 = (15, np.complex128)
    BF16 = (
        16,
        np.float16,
    )  # TODO:numpy does not support bf16 yet, should be handled by backend

    def onnx_type(self):
        return self.value[0]

    def np_type(self):
        return self.value[1]


def find_onnx_type(numpy_type):
    for dtype in DTYPE:
        if dtype.np_type() == numpy_type:
            return dtype.onnx_type()
    raise ValueError(f"No corresponding ONNX type found for numpy type: {numpy_type}")

