"""Static interface for APIs exposed dynamically by the runtime."""
from typing import Any
import tvm
from tvm import tirx
from tvm.backend.cuda import transforms as cuda_transforms

def default_tir_pipeline() -> Any:
    ...

def tirx_pipeline() -> Any:
    ...

def finalize_host_passes() -> Any:
    ...

def finalize_device_passes() -> Any:
    ...

def finalize_device_passes_tirx() -> Any:
    ...
PIPELINE_MAP = {'default': default_tir_pipeline, 'tirx': tirx_pipeline}

def register_tir_pipeline(name: str, pipeline_factory) -> None:
    ...

def get_tir_pipeline(name: str | None=None, **kwargs) -> tuple[tvm.transform.Pass, Any, Any]:
    ...

def get_default_tir_pipeline(target: tvm.target.Target) -> tuple[tvm.transform.Pass, Any, Any]:
    ...
