"""TVM 0.26 script exports, including dynamically registered CUDA namespaces."""

from tvm.backend.cuda.script import CUDANamespace, NVSHMEMNamespace, PTXNamespace
from tvm.tirx.lang.alloc_pool import SMEMPool as _SMEMPool, TMEMPool as TMEMPool
from tvm.tirx.buffer import Buffer as _Buffer
from tvm.tirx.layout import Layout as _Layout

from .parser import *
from .parser import Buffer as Buffer, Ptr as Ptr, prim_func as prim_func
from .parser.entry import inline as inline, macro as macro
from . import tile as tile
from .builder.ir import TensorMap as TensorMap, meta_class as meta_class
from .tile import (
    cluster as cluster,
    cta as cta,
    thread as thread,
    warp as warp,
    warpgroup as warpgroup,
    wg as wg,
)

# Registered by tvm.backend.cuda using register_script_namespace at runtime.
ptx: PTXNamespace
cuda: CUDANamespace
nvshmem: NVSHMEMNamespace

class SMEMPool(_SMEMPool):
    # The implementation accepts layout objects as well as the "default" string.
    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        layout: str | _Layout = "default",
    ) -> _Buffer: ...

# Explicit exports: the runtime builds __all__ dynamically.
from .builder.ir import Cast as Cast
from .builder.ir import address_of as address_of
from .builder.ir import alloc_buffer as alloc_buffer
from .builder.ir import alloc_local as alloc_local
from .builder.ir import attr as attr
from .builder.ir import cta_id as cta_id
from .builder.ir import decl_buffer as decl_buffer
from .builder.ir import device_entry as device_entry
from .builder.ir import evaluate as evaluate
from .builder.ir import exp as exp
from .builder.ir import filter as filter
from .builder.ir import float16 as float16
from .builder.ir import float32 as float32
from .builder.ir import func_attr as func_attr
from .builder.ir import int32 as int32
from .builder.ir import lane_id as lane_id
from .builder.ir import let as let
from .builder.ir import max as max
from .builder.ir import meta_var as meta_var
from .builder.ir import tvm_storage_sync as tvm_storage_sync
from .builder.ir import tvm_warp_shuffle_xor as tvm_warp_shuffle_xor
from .builder.ir import uint32 as uint32
from .builder.ir import unroll as unroll
from .builder.ir import warp_id as warp_id
from .builder.ir import warp_id_in_wg as warp_id_in_wg
from .builder.ir import warpgroup_id as warpgroup_id
