# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util

from ._ax_sys_api import ax_sys_malloc_cached, ax_sys_free
from ._ax_types import *
from ._ax_types_internal import *

__all__: ["ax_engine_get_chip_type", "ax_engine_init", "ax_engine_final", "ax_engine_get_version",
          "ax_engine_get_npu_type", "ax_engine_get_model_type", "ax_engine_load_model", "ax_engine_destroy_handle",
          "ax_engine_get_tool_version", "ax_engine_get_group_count", "ax_engine_get_info", "ax_engine_create_io",
          "ax_engine_destroy_io", "ax_engine_run"]

engine_path = ctypes.util.find_library(engine_name)
assert (engine_path is not None), \
    f"Failed to find library {engine_name}. Please ensure it is installed and in the library path."

engine_lib = ctypes.CDLL(engine_path)
assert (engine_lib is not None), \
    f"Failed to load library {engine_path}. Please ensure it is installed and in the library path."


def _check_func_exists(lib, func_name):
    try:
        ret = getattr(lib, func_name)
        return True if ret else False
    except AttributeError:
        return False


def ax_engine_get_chip_type():
    if not _check_func_exists(engine_lib, "AX_ENGINE_SetAffinity"):
        return ChipType.M57H
    elif not _check_func_exists(engine_lib, "AX_ENGINE_GetTotalOps"):
        return ChipType.MC50
    else:
        return ChipType.MC20E


def ax_engine_init(npu_type: VNPUType) -> int:
    ret = -1
    try:
        engine_lib.AX_ENGINE_Init.restype = AX_S32
        engine_lib.AX_ENGINE_Init.argtypes = [ctypes.POINTER(AX_ENGINE_NPU_ATTR_T)]

        c_attr = AX_ENGINE_NPU_ATTR_T()
        c_attr.eHardMode = npu_type.value

        ret = engine_lib.AX_ENGINE_Init(ctypes.byref(c_attr))
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_engine_final() -> int:
    ret = -1
    try:
        engine_lib.AX_ENGINE_Deinit.restype = AX_S32
        ret = engine_lib.AX_ENGINE_Deinit()
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_engine_get_version() -> str:
    version = None
    try:
        engine_lib.AX_ENGINE_GetVersion.restype = ctypes.c_char_p
        version = engine_lib.AX_ENGINE_GetVersion()
    except Exception as e:
        print(e)
    finally:
        return version.decode("utf-8") if version else ""


def ax_engine_get_npu_type() -> tuple[VNPUType, int]:
    ret = -1
    npu_type = VNPUType.DISABLED

    try:
        c_npu_attr = AX_ENGINE_NPU_ATTR_T()

        engine_lib.AX_ENGINE_GetVNPUAttr.restype = AX_S32
        engine_lib.AX_ENGINE_GetVNPUAttr.argtypes = [ctypes.POINTER(AX_ENGINE_NPU_ATTR_T)]

        ret = engine_lib.AX_ENGINE_GetVNPUAttr(ctypes.byref(c_npu_attr))
        npu_type = VNPUType(c_npu_attr.eHardMode)
    except Exception as e:
        print(e)
    finally:
        return npu_type, ret


def ax_engine_get_model_type(ptr: int, size: int) -> tuple[ModelType, int]:
    ret = -1
    model_type = ModelType.SINGLE

    try:
        c_buffer_ptr = ctypes.cast(ctypes.c_void_p(ptr), ctypes.c_void_p)
        c_buffer_size = ctypes.c_uint32(size)
        c_model_type = ctypes.c_int32()

        engine_lib.AX_ENGINE_GetModelType.restype = AX_S32
        argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]
        engine_lib.AX_ENGINE_GetModelType.argtypes = argtypes

        ret = engine_lib.AX_ENGINE_GetModelType(c_buffer_ptr, c_buffer_size, ctypes.byref(c_model_type))
        model_type = ModelType(c_model_type.value)
    except Exception as e:
        print(e)
    finally:
        return model_type, ret


def ax_engine_load_model(ptr: int, size: int) -> tuple[int, int, int]:
    ret = -1
    handle = 0
    context = 0

    try:
        c_buffer_ptr = ctypes.cast(ctypes.c_void_p(ptr), ctypes.c_void_p)
        c_buffer_size = ctypes.c_uint32(size)
        c_handle = AX_ENGINE_HANDLE()
        c_handle_extra = AX_ENGINE_HANDLE_EXTRA_T()
        c_handle_extra.pName = "PyEngine".encode("utf-8")
        c_handle_extra.nNpuSet = AX_ENGINE_NPU_SET_T(1)

        engine_lib.AX_ENGINE_CreateHandleV2.restype = AX_S32
        argtypes = [ctypes.POINTER(AX_ENGINE_HANDLE), ctypes.c_void_p, ctypes.c_uint32,
                    ctypes.POINTER(AX_ENGINE_HANDLE_EXTRA_T)]
        engine_lib.AX_ENGINE_CreateHandleV2.argtypes = argtypes

        ret = engine_lib.AX_ENGINE_CreateHandleV2(ctypes.byref(c_handle), c_buffer_ptr, c_buffer_size, c_handle_extra)
        handle = c_handle.value

        if ret == 0:
            c_context = AX_ENGINE_CONTEXT_T()
            engine_lib.AX_ENGINE_CreateContextV2.restype = AX_S32
            engine_lib.AX_ENGINE_CreateContextV2.argtypes = [AX_ENGINE_HANDLE, ctypes.POINTER(AX_ENGINE_CONTEXT_T)]
            ret = engine_lib.AX_ENGINE_CreateContextV2(c_handle, ctypes.byref(c_context))
            context = c_context.value
    except Exception as e:
        print(e)
    finally:
        return handle, context, ret


def ax_engine_destroy_handle(handle: int) -> int:
    ret = -1
    try:
        c_handle = ctypes.cast(ctypes.c_void_p(handle), ctypes.c_void_p)

        engine_lib.AX_ENGINE_DestroyHandle.restype = AX_S32
        engine_lib.AX_ENGINE_DestroyHandle.argtypes = [AX_ENGINE_HANDLE]

        ret = engine_lib.AX_ENGINE_DestroyHandle(c_handle)
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_engine_get_tool_version(handle: int) -> str:
    version = None
    try:
        c_handle = ctypes.cast(ctypes.c_void_p(handle), ctypes.c_void_p)

        engine_lib.AX_ENGINE_GetModelToolsVersion.restype = ctypes.c_char_p
        engine_lib.AX_ENGINE_GetModelToolsVersion.argtypes = [AX_ENGINE_HANDLE]
        version = engine_lib.AX_ENGINE_GetModelToolsVersion(c_handle)
    except Exception as e:
        print(e)
    finally:
        return version.decode("utf-8") if version else ""


def ax_engine_get_group_count(handle: int) -> tuple[int, int]:
    ret = -1
    count = 1

    try:
        c_handle = ctypes.cast(ctypes.c_void_p(handle), ctypes.c_void_p)
        c_count = ctypes.c_uint32()

        engine_lib.AX_ENGINE_GetGroupIOInfoCount.restype = AX_S32
        engine_lib.AX_ENGINE_GetGroupIOInfoCount.argtypes = [AX_ENGINE_HANDLE, ctypes.POINTER(ctypes.c_uint32)]

        ret = engine_lib.AX_ENGINE_GetGroupIOInfoCount(c_handle, ctypes.byref(c_count))
        count = c_count.value
    except Exception as e:
        print(e)
    finally:
        return count, ret


def ax_engine_get_info(handle: int) -> tuple[dict, int]:
    ret = -1
    info = dict()

    c_infos = []
    try:
        c_handle = ctypes.cast(ctypes.c_void_p(handle), ctypes.c_void_p)
        c_info = ctypes.POINTER(AX_ENGINE_IO_INFO_T)()

        count, ret = ax_engine_get_group_count(handle)
        if ret != 0:
            raise Exception("Failed to get group count")

        if count == 1:
            engine_lib.AX_ENGINE_GetIOInfo.restype = AX_S32
            argtypes = [AX_ENGINE_HANDLE, ctypes.POINTER(ctypes.POINTER(AX_ENGINE_IO_INFO_T))]
            engine_lib.AX_ENGINE_GetIOInfo.argtypes = argtypes
            ret = engine_lib.AX_ENGINE_GetIOInfo(c_handle, ctypes.byref(c_info))
            if ret != 0:
                raise Exception("Failed to get info")
            c_infos.append(c_info)
        else:
            for i in range(count):
                c_index = ctypes.c_uint32(i)
                engine_lib.AX_ENGINE_GetGroupIOInfo.restype = AX_S32
                argtypes = [AX_ENGINE_HANDLE, ctypes.c_uint32, ctypes.POINTER(AX_ENGINE_IO_INFO_T)]
                engine_lib.AX_ENGINE_GetGroupIOInfo.argtypes = argtypes
                ret = engine_lib.AX_ENGINE_GetGroupIOInfo(c_handle, c_index, ctypes.byref(c_info))
                if ret != 0:
                    raise Exception("Failed to get info")
                c_infos.append(c_info)
    except Exception as e:
        print(e)
    finally:
        inputs = []
        for i in range(c_infos[0].contents.nInputSize):
            shapes = []
            sizes = []
            for k in range(len(c_infos)):
                one_shape = []
                for j in range(c_infos[k].contents.pInputs[i].nShapeSize):
                    one_shape.append(c_infos[k].contents.pInputs[i].pShape[j])
                one_size = c_infos[k].contents.pInputs[i].nSize
                shapes.append(one_shape)
                sizes.append(one_size)
            inputs.append({
                "name": c_infos[0].contents.pInputs[i].pName.decode("utf-8"),
                "shapes": shapes,
                "sizes": sizes,
                "layout": c_infos[0].contents.pInputs[i].eLayout,
                "dtype": c_infos[0].contents.pInputs[i].eDataType,
            })
        outputs = []
        for i in range(c_infos[0].contents.nOutputSize):
            shapes = []
            sizes = []
            for k in range(len(c_infos)):
                one_shape = []
                for j in range(c_infos[k].contents.pOutputs[i].nShapeSize):
                    one_shape.append(c_infos[k].contents.pOutputs[i].pShape[j])
                one_size = c_infos[k].contents.pOutputs[i].nSize
                shapes.append(one_shape)
                sizes.append(one_size)
            outputs.append({
                "name": c_infos[0].contents.pOutputs[i].pName.decode("utf-8"),
                "shapes": shapes,
                "sizes": sizes,
                "layout": c_infos[0].contents.pOutputs[i].eLayout,
                "dtype": c_infos[0].contents.pOutputs[i].eDataType,
            })
        info["inputs"] = inputs
        info["outputs"] = outputs
        return info, ret


def ax_engine_create_io(info: dict) -> dict or None:
    io = dict()
    io["inputs"] = []
    io["outputs"] = []
    for i in range(len(info["inputs"])):
        max_size = 0
        for one_size in info["inputs"][i]["sizes"]:
            max_size = max(max_size, one_size)
        phy, vir, ret = ax_sys_malloc_cached(max_size)
        if ret != 0:
            return None
        else:
            io["inputs"].append({
                "phy": phy,
                "vir": vir,
                "size": max_size,
            })
    for i in range(len(info["outputs"])):
        max_size = 0
        for one_size in info["outputs"][i]["sizes"]:
            max_size = max(max_size, one_size)
        phy, vir, ret = ax_sys_malloc_cached(max_size)
        if ret != 0:
            return None
        else:
            io["outputs"].append({
                "phy": phy,
                "vir": vir,
                "size": max_size,
            })
    return io


def ax_engine_destroy_io(io: dict) -> int:
    ret = -1
    try:
        for i in range(len(io["inputs"])):
            if io["inputs"][i]["vir"]:
                ax_sys_free(io["inputs"][i]["phy"], io["inputs"][i]["vir"])
        for i in range(len(io["outputs"])):
            if io["outputs"][i]["vir"]:
                ax_sys_free(io["outputs"][i]["phy"], io["outputs"][i]["vir"])
        ret = 0
    except Exception as e:
        print(e)
    finally:
        return ret


def ax_engine_run(handle: int, context: int, index: int, io: dict) -> int:
    ret = -1
    try:
        c_handle = ctypes.cast(ctypes.c_void_p(handle), AX_ENGINE_HANDLE)
        c_context = ctypes.cast(ctypes.c_void_p(context), AX_ENGINE_CONTEXT_T)
        c_index = ctypes.c_uint32(index)
        c_io = AX_ENGINE_IO_T()

        c_io.nInputSize = len(io["inputs"])
        c_io.pInputs = (AX_ENGINE_IO_BUFFER_T * c_io.nInputSize)()
        for i in range(c_io.nInputSize):
            c_io.pInputs[i].phyAddr = io["inputs"][i]["phy"]
            c_io.pInputs[i].pVirAddr = io["inputs"][i]["vir"]
            c_io.pInputs[i].nSize = io["inputs"][i]["size"]
            c_io.pInputs[i].nStrideSize = 0
            c_io.pInputs[i].pStride = None
        c_io.nOutputSize = len(io["outputs"])
        c_io.pOutputs = (AX_ENGINE_IO_BUFFER_T * c_io.nOutputSize)()
        for i in range(c_io.nOutputSize):
            c_io.pOutputs[i].phyAddr = io["outputs"][i]["phy"]
            c_io.pOutputs[i].pVirAddr = io["outputs"][i]["vir"]
            c_io.pOutputs[i].nSize = io["outputs"][i]["size"]
            c_io.pOutputs[i].nStrideSize = 0
            c_io.pOutputs[i].pStride = None
        c_io.nMaxBatchSize = 1
        c_io.pIoSetting = None

        if index == 0:
            engine_lib.AX_ENGINE_RunSyncV2.restype = AX_S32
            argtypes = [AX_ENGINE_HANDLE, AX_ENGINE_CONTEXT_T, ctypes.POINTER(AX_ENGINE_IO_T)]
            engine_lib.AX_ENGINE_RunSyncV2.argtypes = argtypes
            ret = engine_lib.AX_ENGINE_RunSyncV2(c_handle, c_context, ctypes.byref(c_io))
        else:
            engine_lib.AX_ENGINE_RunGroupIOSync.restype = AX_S32
            argtypes = [AX_ENGINE_HANDLE, AX_ENGINE_CONTEXT_T, ctypes.c_uint32, ctypes.POINTER(AX_ENGINE_IO_T)]
            engine_lib.AX_ENGINE_RunGroupIOSync.argtypes = argtypes
            ret = engine_lib.AX_ENGINE_RunGroupIOSync(c_handle, c_context, c_index, ctypes.byref(c_io))
    except Exception as e:
        print(e)
    finally:
        return ret
