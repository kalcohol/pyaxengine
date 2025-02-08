# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util

import numpy as np

from ._ax_types_internal import *

__all__: ["ax_sys_init", "ax_sys_final", "ax_sys_malloc", "ax_sys_malloc_cached", "ax_sys_free", "ax_sys_flush",
          "ax_sys_invalidate", "ax_sys_copy_from_numpy", "ax_sys_copy_to_numpy"]

sys_path = ctypes.util.find_library(sys_name)
assert (sys_path is not None), \
    f"Failed to find library {sys_name}. Please ensure it is installed and in the library path."

sys_lib = ctypes.CDLL(sys_path)
assert (sys_lib is not None), \
    f"Failed to load library {sys_path}. Please ensure it is installed and in the library path."


def ax_sys_init() -> int:
    ret = -1
    try:
        sys_lib.AX_SYS_Init.restype = AX_S32
        ret = sys_lib.AX_SYS_Init()
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_final() -> int:
    ret = -1
    try:
        sys_lib.AX_SYS_Deinit.restype = AX_S32
        ret = sys_lib.AX_SYS_Deinit()
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_malloc_proxy(size: int, cached: bool) -> tuple[int, int, int]:
    ret = -1
    c_phy_addr = AX_U64(0)
    c_vir_addr = ctypes.c_void_p(0)

    try:
        sys_lib.AX_SYS_MemAlloc.restype = AX_S32
        sys_lib.AX_SYS_MemAllocCached.restype = AX_S32

        argtypes = [ctypes.POINTER(AX_U64), ctypes.POINTER(ctypes.c_void_p), AX_U32, AX_U32, ctypes.c_char_p]

        sys_lib.AX_SYS_MemAlloc.argtypes = argtypes
        sys_lib.AX_SYS_MemAllocCached.argtypes = argtypes

        c_size = AX_U32(size)
        c_align = AX_U32(sys_align_size)
        c_token = ctypes.c_char_p(sys_token_name.encode('utf-8'))

        if cached:
            ax_sys_malloc_func = sys_lib.AX_SYS_MemAllocCached
        else:
            ax_sys_malloc_func = sys_lib.AX_SYS_MemAlloc

        ret = ax_sys_malloc_func(ctypes.byref(c_phy_addr), ctypes.byref(c_vir_addr), c_size, c_align, c_token)
    except Exception as e:
        print(e)
    finally:
        return c_phy_addr.value, c_vir_addr.value, ret


def ax_sys_malloc(size: int) -> tuple[int, int, int]:
    return ax_sys_malloc_proxy(size, False)


def ax_sys_malloc_cached(size: int) -> tuple[int, int, int]:
    return ax_sys_malloc_proxy(size, True)


def ax_sys_free(phy: int, vir: int) -> int:
    ret = -1
    try:
        sys_lib.AX_SYS_MemFree.restype = AX_S32
        sys_lib.AX_SYS_MemFree.argtypes = [AX_U64, ctypes.c_void_p]
        c_phy_addr = AX_U64(phy)
        c_vir_addr = ctypes.c_void_p(vir)

        ret = sys_lib.AX_SYS_MemFree(c_phy_addr, c_vir_addr)
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_flush(phy: int, vir: int, size: int) -> int:
    ret = -1
    try:
        sys_lib.AX_SYS_MflushCache.restype = AX_S32
        sys_lib.AX_SYS_MflushCache.argtypes = [AX_U64, ctypes.c_void_p, AX_U32]
        c_phy_addr = AX_U64(phy)
        c_vir_addr = ctypes.c_void_p(vir)
        c_size = AX_U32(size)

        ret = sys_lib.AX_SYS_MflushCache(c_phy_addr, c_vir_addr, c_size)
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_invalidate(phy: int, vir: int, size: int) -> int:
    ret = -1
    try:
        sys_lib.AX_SYS_MinvalidateCache.restype = AX_S32
        sys_lib.AX_SYS_MinvalidateCache.argtypes = [AX_U64, ctypes.c_void_p, AX_U32]
        c_phy_addr = AX_U64(phy)
        c_vir_addr = ctypes.c_void_p(vir)
        c_size = AX_U32(size)

        ret = sys_lib.AX_SYS_MinvalidateCache(c_phy_addr, c_vir_addr, c_size)
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_copy_from_numpy(phy: int, vir: int, data: np.ndarray) -> int:
    ret = -1
    try:
        if not (data.flags.c_contiguous or data.flags.f_contiguous):
            data = np.ascontiguousarray(data)
        ctypes.memmove(vir, data.ctypes.data, data.nbytes)
        ret = ax_sys_flush(phy, vir, data.nbytes)
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_sys_copy_to_numpy(phy: int, vir: int, shape, dtype) -> np.ndarray | None:
    try:
        ptr = ctypes.cast(vir, ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype)))
        numpy_array = np.ctypeslib.as_array(ptr, shape)
        ret = ax_sys_invalidate(phy, vir, numpy_array.nbytes)
        if ret != 0:
            return None
        return numpy_array
    except Exception as e:
        print(e)
