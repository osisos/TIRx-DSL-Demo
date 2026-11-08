import multiprocessing
import sys
import os
from tvm_ffi import register_object as register_object, register_global_func as register_global_func, get_global_func as get_global_func
from .libinfo import __version__ as __version__
from .base import _RUNTIME_ONLY as _RUNTIME_ONLY
from .runtime import Object as Object
from .runtime._tensor import device as device, device_from_target as device_from_target, cpu as cpu, cuda as cuda, opencl as opencl, vulkan as vulkan, metal as metal
from .runtime._tensor import vpi as vpi, rocm as rocm, ext_dev as ext_dev, hexagon as hexagon
from .runtime import DataType as DataType, DataTypeCode as DataTypeCode
from . import error as error
from .ir import IRModule as IRModule
from .ir import transform as transform
from .ir import instrument as instrument
from . import ir as ir
from . import script as script
from . import tirx as tirx
from . import backend as backend
from . import target as target
from . import te as te
from .driver import build as build, compile as compile
from . import arith as arith
from . import support as support
from .support import rocm as _rocm, nvcc as _nvcc

def _should_print_backtrace() -> Any:
    ...

def tvm_wrap_excepthook(exception_hook) -> Any:
    ...
sys.excepthook = tvm_wrap_excepthook(sys.excepthook)
from .backend._autoload_backends import _autoload_backends as _autoload_backends
from . import relax as relax
