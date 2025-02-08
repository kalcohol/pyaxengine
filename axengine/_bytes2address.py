# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes
from ctypes import c_char
from ctypes import Array


def bytes_to_address(data : bytes) -> tuple[Array[c_char], int, int] | tuple[int, int]:
    size = len(data)
    buffer = ctypes.create_string_buffer(data, size)
    ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))
    return buffer, ctypes.addressof(ptr.contents), size
