"""Static interface for APIs exposed dynamically by the runtime."""
from typing import Any
import ctypes
import warnings
from typing import Optional
import numpy as np
import tvm_ffi
from tvm_ffi import DLDeviceType, device
import tvm
from tvm.runtime import Device
from . import _ffi_api

def from_dlpack(ext_tensor) -> Any:
    ...

@tvm_ffi.register_object('ffi.Tensor')
class Tensor(tvm_ffi.core.Tensor):
    __slots__ = ()

    def __setitem__(self, in_slice, value) -> Any:
        ...

    def copyfrom(self, source_array) -> Any:
        ...

    def __repr__(self) -> Any:
        ...

    def __str__(self) -> Any:
        ...

    def numpy(self) -> Any:
        ...

    def copyto(self, target, mem_scope=None) -> Any:
        ...

    def _copyto(self, target_nd) -> Any:
        ...

    def _create_view(self, shape, dtype: str | None=None, relative_byte_offset: int=0) -> Any:
        ...

def empty(shape, dtype='float32', device=None, mem_scope=None) -> Any:
    ...

def tensor(arr, device=None, mem_scope=None) -> Any:
    ...

def device_from_target(target, index=None) -> Device:
    ...

def cpu(dev_id=0) -> Device:
    ...

def cuda(dev_id=0) -> Device:
    ...

def rocm(dev_id=0) -> Device:
    ...

def opencl(dev_id=0) -> Device:
    ...

def metal(dev_id=0) -> Device:
    ...

def vpi(dev_id=0) -> Device:
    ...

def vulkan(dev_id=0) -> Device:
    ...

def ext_dev(dev_id=0) -> Device:
    ...

def hexagon(dev_id=0) -> Device:
    ...

def webgpu(dev_id=0) -> Device:
    ...

def device(dev_type, dev_id=0) -> Device:
    ...
