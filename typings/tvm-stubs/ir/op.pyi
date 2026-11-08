"""Static interface for APIs exposed dynamically by the runtime."""
from typing import Any
import tvm_ffi
from . import _ffi_api
from .expr import Expr

@tvm_ffi.register_object('ir.Op')
class Op(Expr):
    name: str

    def __init__(self) -> Any:
        ...

    @staticmethod
    def get(op_name) -> Any:
        ...

    def get_attr(self, attr_name) -> Any:
        ...

    def has_attr(self, attr_name) -> Any:
        ...

    def set_attr(self, attr_name, value, plevel=10) -> Any:
        ...

    def reset_attr(self, attr_name) -> Any:
        ...

    def add_argument(self, name, type, description) -> Any:
        ...

    def set_support_level(self, level) -> Any:
        ...

    def set_num_inputs(self, n) -> Any:
        ...

    def set_attrs_type_key(self, key) -> Any:
        ...

    @staticmethod
    def list_op_names() -> Any:
        ...

def register_op_attr(op_name, attr_key, value=None, level=10) -> Any:
    ...

def register_intrin_lowering(op_name, target, *, f=None, level=10) -> Any:
    ...
