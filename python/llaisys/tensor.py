from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    @staticmethod
    def from_torch(torch_tensor, dtype: DataType = DataType.BF16, device: DeviceType = DeviceType.CPU):
        """Create Tensor from PyTorch tensor, handling bfloat16 conversion."""
        import torch
        import numpy as np

        # Get shape
        shape = list(torch_tensor.shape)

        # Handle bfloat16 by converting to float32 for data transfer
        if torch_tensor.dtype == torch.bfloat16:
            # Convert to float32 for transfer
            data_tensor = torch_tensor.float()  # Convert to float32
            # Create numpy array
            np_array = data_tensor.numpy().astype(np.float32)
            # Use float32 for storage since bfloat16 isn't natively supported
            target_dtype = DataType.F32
        elif torch_tensor.dtype == torch.float16:
            np_array = torch_tensor.numpy().astype(np.float16)
            target_dtype = DataType.F16
        elif torch_tensor.dtype == torch.float32:
            np_array = torch_tensor.numpy().astype(np.float32)
            target_dtype = DataType.F32
        elif torch_tensor.dtype == torch.int64:
            np_array = torch_tensor.numpy().astype(np.int64)
            target_dtype = DataType.I64
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_tensor.dtype}")

        # Create LLAISYS tensor
        llaisys_tensor = Tensor(shape=shape, dtype=target_dtype, device=device)

        # Load data
        llaisys_tensor.load(np_array.ctypes.data_as(c_void_p))

        return llaisys_tensor

    @staticmethod
    def from_numpy(np_array, dtype: DataType = None, device: DeviceType = DeviceType.CPU):
        """Create Tensor from numpy array."""
        # Infer dtype if not specified
        if dtype is None:
            dtype_map = {
                np.float32: DataType.F32,
                np.float16: DataType.F16,
                np.int64: DataType.I64,
                np.int32: DataType.I32,
                np.float64: DataType.F64,
            }
            dtype = dtype_map.get(np_array.dtype, DataType.F32)

        shape = list(np_array.shape)

        # Create LLAISYS tensor
        llaisys_tensor = Tensor(shape=shape, dtype=dtype, device=device)

        # Load data
        llaisys_tensor.load(np_array.ctypes.data_as(c_void_p))

        return llaisys_tensor

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape: int) -> llaisysTensor_t:
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )
