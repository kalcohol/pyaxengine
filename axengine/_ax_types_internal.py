# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes
import platform

# library names
sys_name = "ax_sys"
engine_name = "ax_engine"

# pre-defined values
sys_align_size = 4096
sys_token_name = "PyAxEngine"

# common bsp types
AX_S32 = ctypes.c_int
AX_U32 = ctypes.c_uint
AX_U64 = ctypes.c_ulonglong
AX_S8 = ctypes.c_char
AX_VOID = None

# ax engine types
AX_ENGINE_HANDLE = ctypes.c_void_p
AX_ENGINE_CONTEXT_T = ctypes.c_void_p
AX_ENGINE_EXECUTION_CONTEXT = ctypes.c_void_p
AX_ENGINE_NPU_SET_T = ctypes.c_uint32
AX_ENGINE_TENSOR_LAYOUT_T = ctypes.c_int32
AX_ENGINE_MEMORY_TYPE_T = ctypes.c_int32
AX_ENGINE_DATA_TYPE_T = ctypes.c_int32
AX_ENGINE_COLOR_SPACE_T = ctypes.c_int32

# ax engine data types
AX_ENGINE_DT_UNKNOWN = 0
AX_ENGINE_DT_UINT8 = 1
AX_ENGINE_DT_UINT16 = 2
AX_ENGINE_DT_FLOAT32 = 3
AX_ENGINE_DT_SINT16 = 4
AX_ENGINE_DT_SINT8 = 5
AX_ENGINE_DT_SINT32 = 6
AX_ENGINE_DT_UINT32 = 7
AX_ENGINE_DT_FLOAT64 = 8
AX_ENGINE_DT_BFLOAT16 = 9
AX_ENGINE_DT_UINT10_PACKED = 100
AX_ENGINE_DT_UINT12_PACKED = 101
AX_ENGINE_DT_UINT14_PACKED = 102
AX_ENGINE_DT_UINT16_PACKED = 103


# ax engine visual npu type enums
class AX_ENGINE_NPU_ATTR_T(ctypes.Structure):
    _fields_ = [
        ("eHardMode", ctypes.c_int),
        ("reserve", ctypes.c_uint * 8),
    ]


class AX_ENGINE_HANDLE_EXTRA_T(ctypes.Structure):
    _fields_ = [
        ("nNpuSet", AX_ENGINE_NPU_SET_T),
        ("pName", ctypes.c_char_p),
        ("reserve", ctypes.c_uint32 * 8)
    ]


class AX_ENGINE_IOMETA_EX_T(ctypes.Structure):
    _fields_ = [
        ("eColorSpace", AX_ENGINE_COLOR_SPACE_T),
        ("u64Reserved", ctypes.c_uint64 * 18)
    ]


class AX_ENGINE_IO_SETTING_T(ctypes.Structure):
    _fields_ = [
        ("nWbtIndex", ctypes.c_uint32),
        ("u64Reserved", ctypes.c_uint64 * 7)
    ]


# check architecture, 32bit or 64bit
arch = platform.architecture()[0]

if arch == "64bit":
    _meta_reserved_size = 9
    _io_info_reserved_size = 11
    _io_buffer_reserved_size = 11
    _io_reserved_size = 11
else:
    _meta_reserved_size = 11
    _io_info_reserved_size = 13
    _io_buffer_reserved_size = 13
    _io_reserved_size = 13


class AX_ENGINE_IOMETA_T(ctypes.Structure):
    _fields_ = [
        ("pName", ctypes.c_char_p),
        ("pShape", ctypes.POINTER(ctypes.c_int32)),
        ("nShapeSize", ctypes.c_uint8),
        ("eLayout", AX_ENGINE_TENSOR_LAYOUT_T),
        ("eMemoryType", AX_ENGINE_MEMORY_TYPE_T),
        ("eDataType", AX_ENGINE_DATA_TYPE_T),
        ("pExtraMeta", ctypes.POINTER(AX_ENGINE_IOMETA_EX_T)),
        ("nSize", ctypes.c_uint32),
        ("nQuantizationValue", ctypes.c_uint32),
        ("pStride", ctypes.POINTER(ctypes.c_int32)),
        ("u64Reserved", ctypes.c_uint64 * _meta_reserved_size)
    ]


class AX_ENGINE_IO_INFO_T(ctypes.Structure):
    _fields_ = [
        ("pInputs", ctypes.POINTER(AX_ENGINE_IOMETA_T)),
        ("nInputSize", ctypes.c_uint32),
        ("pOutputs", ctypes.POINTER(AX_ENGINE_IOMETA_T)),
        ("nOutputSize", ctypes.c_uint32),
        ("nMaxBatchSize", ctypes.c_uint32),
        ("bDynamicBatchSize", ctypes.c_int32),
        ("u64Reserved", ctypes.c_uint64 * _io_info_reserved_size)
    ]


class AX_ENGINE_IO_BUFFER_T(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_uint64),
        ("pVirAddr", ctypes.c_void_p),
        ("nSize", ctypes.c_uint32),
        ("pStride", ctypes.POINTER(ctypes.c_int32)),
        ("nStrideSize", ctypes.c_uint8),
        ("u64Reserved", ctypes.c_uint64 * _io_buffer_reserved_size)
    ]


class AX_ENGINE_IO_T(ctypes.Structure):
    if arch == "64bit":
        _fields_ = [
            ("pInputs", ctypes.POINTER(AX_ENGINE_IO_BUFFER_T)),
            ("nInputSize", ctypes.c_uint32),
            ("pOutputs", ctypes.POINTER(AX_ENGINE_IO_BUFFER_T)),
            ("nOutputSize", ctypes.c_uint32),
            ("nBatchSize", ctypes.c_uint32),
            ("pIoSetting", ctypes.POINTER(AX_ENGINE_IO_SETTING_T)),
            ("nParallelRun", ctypes.c_uint32),
            ("u64Reserved", ctypes.c_uint64 * _io_reserved_size)
        ]
    else:
        _fields_ = [
            ("pInputs", ctypes.POINTER(AX_ENGINE_IO_BUFFER_T)),
            ("nInputSize", ctypes.c_uint32),
            ("pOutputs", ctypes.POINTER(AX_ENGINE_IO_BUFFER_T)),
            ("nOutputSize", ctypes.c_uint32),
            ("nBatchSize", ctypes.c_uint32),
            ("pIoSetting", ctypes.POINTER(AX_ENGINE_IO_SETTING_T)),
            ("u64Reserved", ctypes.c_uint64 * _io_reserved_size)
        ]
