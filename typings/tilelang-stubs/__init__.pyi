"""Public TileLang entry points exposed by the full runtime import."""
import tvm as tvm
from . import analysis as analysis
from .engine import lower as lower, register_cuda_postproc as register_cuda_postproc, register_hip_postproc as register_hip_postproc, register_c_postproc as register_c_postproc
from .jit import jit as jit, JITKernel as JITKernel, compile as compile, par_compile as par_compile
from .env import disable_cache as disable_cache, enable_cache as enable_cache, is_cache_enabled as is_cache_enabled

def set_log_level(level: str | int) -> None: ...
