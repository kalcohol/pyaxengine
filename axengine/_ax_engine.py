# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import atexit
import os
from typing import Any

import ml_dtypes as mldt

from ._ax_engine_api import *
from ._ax_sys_api import *
from ._base_session import Session, SessionOptions
from ._bytes2address import bytes_to_address
from ._node import NodeArg

__all__: ["AXEngineSession"]

_is_sys_initialized = False
_is_engine_initialized = False


def _transform_dtype(dtype):
    if dtype == AX_ENGINE_DT_UINT8:
        return np.dtype(np.uint8)
    elif dtype == AX_ENGINE_DT_SINT8:
        return np.dtype(np.int8)
    elif dtype == AX_ENGINE_DT_UINT16:
        return np.dtype(np.uint16)
    elif dtype == AX_ENGINE_DT_SINT16:
        return np.dtype(np.int16)
    elif dtype == AX_ENGINE_DT_UINT32:
        return np.dtype(np.uint32)
    elif dtype == AX_ENGINE_DT_SINT32:
        return np.dtype(np.int32)
    elif dtype == AX_ENGINE_DT_FLOAT32:
        return np.dtype(np.float32)
    elif dtype == AX_ENGINE_DT_BFLOAT16:
        return np.dtype(mldt.bfloat16)
    else:
        raise ValueError(f"Unsupported data type '{dtype}'.")


def _initialize_engine():
    global _is_sys_initialized, _is_engine_initialized

    ret = ax_sys_init()
    if ret != 0:
        raise RuntimeError("Failed to initialize ax sys.")
    _is_sys_initialized = True

    # disabled mode by default
    npu_type, ret = ax_engine_get_npu_type()
    if 0 != ret:
        # this means the NPU was not initialized
        npu_type = VNPUType.DISABLED
    ret = ax_engine_init(npu_type)
    if ret != 0:
        raise RuntimeError("Failed to initialize ax sys engine.")
    _is_engine_initialized = True

    print(f"[INFO] Chip type: {ax_engine_get_chip_type()}")
    print(f"[INFO] VNPU type: {npu_type}")
    print(f"[INFO] Engine version: {ax_engine_get_version()}")


def _finalize_engine():
    global _is_sys_initialized, _is_engine_initialized

    if _is_engine_initialized:
        ax_engine_final()
    if _is_sys_initialized:
        ax_sys_final()


_initialize_engine()
atexit.register(_finalize_engine)


class AXEngineSession(Session):
    def __init__(
            self,
            path_or_bytes: str | bytes | os.PathLike,
            sess_options: SessionOptions | None = None,
            provider_options: dict[Any, Any] | None = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self._chip_type = ax_engine_get_chip_type()
        self._vnpu_type = ax_engine_get_npu_type()[0]

        # handle, context, info, io
        self._handle = 0
        self._io = None

        # model buffer, almost copied from onnx runtime
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_name = os.path.splitext(os.path.basename(path_or_bytes))[0]
            with open(path_or_bytes, "rb") as f:
                data = f.read()
                self._model_buffer, self._model_buffer_ptr, self._model_buffer_size = bytes_to_address(data)
        elif isinstance(path_or_bytes, bytes):
            self._model_buffer, self._model_buffer_ptr, self._model_buffer_size = bytes_to_address(path_or_bytes)
        else:
            raise TypeError(f"Unable to load model from type '{type(path_or_bytes)}'")

        # get model type
        self._model_type, ret = ax_engine_get_model_type(self._model_buffer_ptr, self._model_buffer_size)
        if self._chip_type is ChipType.MC20E:
            if self._model_type is ModelType.FULL:
                print(f"[INFO] Model type: {self._model_type.value} (full core)")
            if self._model_type is ModelType.HALF:
                print(f"[INFO] Model type: {self._model_type.value} (half core)")
        if self._chip_type is ChipType.MC50:
            if self._model_type is ModelType.SINGLE:
                print(f"[INFO] Model type: {self._model_type.value} (single core)")
            if self._model_type is ModelType.DUAL:
                print(f"[INFO] Model type: {self._model_type.value} (dual core)")
            if self._model_type is ModelType.TRIPLE:
                print(f"[INFO] Model type: {self._model_type.value} (triple core)")
        if self._chip_type is ChipType.M57H:
            print(f"[INFO] Model type: {self._model_type.value} (single core)")

        # check model type
        if self._chip_type is ChipType.MC50:
            # all types (single or dual or triple) of model are allowed in vnpu mode disabled
            # only single core model is allowed in vnpu mode enabled
            # only triple core model is NOT allowed in vnpu mode big-little or little-big
            if self._vnpu_type is VNPUType.ENABLED:
                if self._model_type is not ModelType.SINGLE:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
            if (
                    self._vnpu_type is VNPUType.BIG_LITTLE
                    or self._vnpu_type is VNPUType.LITTLE_BIG
            ):
                if self._model_type is ModelType.TRIPLE:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
        if self._chip_type is ChipType.MC20E:
            # all types of full or half core model are allowed in vnpu mode disabled
            # only half core model is allowed in vnpu mode enabled
            if self._vnpu_type is VNPUType.ENABLED:
                if self._model_type is ModelType.FULL:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
        # if self._chip_type is ChipType.M57H:
        # there only one type of model will be compiled, so no need to check

        # load model
        self._handle, self._context, ret = ax_engine_load_model(self._model_buffer_ptr, self._model_buffer_size)
        if 0 != ret:
            raise RuntimeError("Failed to load model.")
        print(f"[INFO] Compiler version: {ax_engine_get_tool_version(self._handle)}")

        # get shape group count
        try:
            self._shape_count, ret = ax_engine_get_group_count(self._handle)
            if 0 != ret:
                raise RuntimeError("Failed to get model shape group count.")
        except AttributeError as e:
            print(f"[WARNING] {e}")
            self._shape_count = 1

        # get model shape
        self._info, ret = ax_engine_get_info(self._handle)
        if 0 != ret:
            raise RuntimeError("Failed to get model info.")
        self._io = ax_engine_create_io(self._info)
        if self._io is None:
            raise RuntimeError("Failed to create model io.")
        self._inputs = self._get_inputs()
        self._outputs = self._get_outputs()

    def __del__(self):
        self._unload()

    def _unload(self):
        if self._handle is not None:
            ax_engine_destroy_io(self._io)
            ax_engine_destroy_handle(self._handle)
            self._handle = None

    def _get_inputs(self):
        inputs = []
        for shape_index in range(self._shape_count):
            one_input_group = []
            for input_index, one_input in enumerate(self._info["inputs"]):
                current_group_shape = one_input["shapes"][shape_index]
                one_input_group.append(
                    NodeArg(
                        one_input["name"],
                        _transform_dtype(one_input["dtype"]),
                        current_group_shape,
                    )
                )
            inputs.append(one_input_group)
        return inputs

    def _get_outputs(self):
        outputs = []
        for shape_index in range(self._shape_count):
            one_output_group = []
            for output_index, one_output in enumerate(self._info["outputs"]):
                current_group_shape = one_output["shapes"][shape_index]
                one_output_group.append(
                    NodeArg(
                        one_output["name"],
                        _transform_dtype(one_output["dtype"]),
                        current_group_shape,
                    )
                )
            outputs.append(one_output_group)
        return outputs

    def run(
            self,
            output_names: list[str],
            input_feed: dict[str, np.ndarray],
            run_options=None,
            shape_group: int = 0
    ):
        self._validate_input(input_feed)
        self._validate_output(output_names)

        if None is output_names:
            output_names = [o.name for o in self.get_outputs()]

        # fill model io
        for key, npy in input_feed.items():
            for i, one in enumerate(self.get_inputs()):
                if one.name == key:
                    assert (
                            list(one.shape) == list(npy.shape) and one.dtype == npy.dtype
                    ), (f"model inputs({key}) expect shape {one.shape} and dtype {one.dtype}, "
                        f"however gets input with shape {npy.shape} and dtype {npy.dtype}")

                    # copy input data to model io
                    this_input = self._io["inputs"][i]
                    ret = ax_sys_copy_from_numpy(this_input["phy"], this_input["vir"], npy)
                    if 0 != ret:
                        raise RuntimeError("Failed to copy input data to model io.")
                    break

        # execute model
        ret = ax_engine_run(self._handle, self._context, shape_group, self._io)

        # flush output
        outputs = []
        if 0 == ret:
            for i in range(len(self.get_outputs())):
                this_output = self._io["outputs"][i]
                npy = ax_sys_copy_to_numpy(this_output["phy"], this_output["vir"],
                                           self.get_outputs()[i].shape, self.get_outputs()[i].dtype)
                if npy is None:
                    raise RuntimeError("Failed to copy output data from model io.")
                name = self.get_outputs()[i].name
                if name in output_names:
                    outputs.append(npy)
            return outputs
        else:
            raise RuntimeError("Failed to run model.")
