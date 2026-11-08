"""Static interface for APIs exposed dynamically by the runtime."""
from typing import Any
from numbers import Number
import tvm_ffi
import tvm
from ..runtime import Object, Scriptable
from . import _ffi_api, _overload_prim_expr, _tensor_expr_overload
from .base import Node, Span

@tvm_ffi.register_object('ir.Expr')
class Expr(Node):
    span: Span | None
    ty: 'tvm.ir.Type'

def is_prim_expr(value: object) -> bool:
    ...

def is_prim_var(value: object) -> bool:
    ...

@tvm_ffi.register_object('ir.GlobalVar')
class GlobalVar(Expr):
    name_hint: str

    def __init__(self, name_hint: str) -> Any:
        ...

    def __call__(self, *args: Expr) -> Expr:
        ...

class _ExprWithOp(Expr, Scriptable):
    __hash__ = Expr.__hash__

    def expr_ty(self) -> Any:
        ...

    def __add__(self, other) -> Any:
        ...

    def __radd__(self, other) -> Any:
        ...

    def __sub__(self, other) -> Any:
        ...

    def __rsub__(self, other) -> Any:
        ...

    def __mul__(self, other) -> Any:
        ...

    def __rmul__(self, other) -> Any:
        ...

    def __div__(self, other) -> Any:
        ...

    def __rdiv__(self, other) -> Any:
        ...

    def __truediv__(self, other) -> Any:
        ...

    def __rtruediv__(self, other) -> Any:
        ...

    def __floordiv__(self, other) -> Any:
        ...

    def __rfloordiv__(self, other) -> Any:
        ...

    def __mod__(self, other) -> Any:
        ...

    def __rmod__(self, other) -> Any:
        ...

    def __pow__(self, other) -> Any:
        ...

    def __rpow__(self, other) -> Any:
        ...

    def __neg__(self) -> Any:
        ...

    def __lshift__(self, other) -> Any:
        ...

    def __rlshift__(self, other) -> Any:
        ...

    def __rshift__(self, other) -> Any:
        ...

    def __rrshift__(self, other) -> Any:
        ...

    def __and__(self, other) -> Any:
        ...

    def __rand__(self, other) -> Any:
        ...

    def __or__(self, other) -> Any:
        ...

    def __ror__(self, other) -> Any:
        ...

    def __xor__(self, other) -> Any:
        ...

    def __rxor__(self, other) -> Any:
        ...

    def __invert__(self) -> Any:
        ...

    def __lt__(self, other) -> Any:
        ...

    def __le__(self, other) -> Any:
        ...

    def __eq__(self, other) -> Any:
        ...

    def __ne__(self, other) -> Any:
        ...

    def __gt__(self, other) -> Any:
        ...

    def __ge__(self, other) -> Any:
        ...

    def __nonzero__(self) -> Any:
        ...

    def __bool__(self) -> bool:
        ...

    def equal(self, other, span=None) -> Any:
        ...

    def astype(self, dtype, span=None) -> Any:
        ...

    def __call__(self, *args, attrs=None) -> Any:
        ...

    def __getitem__(self, index) -> Any:
        ...

@tvm_ffi.register_object('ir.Call')
class Call(_ExprWithOp):
    op: Expr
    args: list[Expr]
    attrs: 'tvm.ir.Attrs | None'
    ty_args: list['tvm.ir.Type']
    span: Span | None

    def __init__(self, op: Expr | str, args: list[Expr] | tuple[Expr, ...], attrs: 'tvm.ir.Attrs | dict | None'=None, ty_args: list['tvm.ir.Type'] | tuple['tvm.ir.Type', ...] | None=None, span: Span | None=None, ret_ty: 'tvm.ir.Type | str | None'=None) -> None:
        ...

@tvm_ffi.register_object('ir.Var')
class Var(_ExprWithOp):
    name: str
    span: Span | None

    def __init__(self, name: str | None=None, ty: 'tvm.ir.Type | str | None'=None, span: Span | None=None, *, name_hint: str | None=None) -> None:
        ...

@tvm_ffi.register_object('ir.Range')
class Range(Node, Scriptable):
    min: Expr
    extent: Expr
    span: Span | None

    def __init__(self, begin: Expr, end: Expr | None=None, span: Span | None=None) -> None:
        ...

    @staticmethod
    def from_min_extent(min_value: Expr, extent: Expr, span: Span | None=None) -> 'Range':
        ...

    def __eq__(self, other: Object) -> bool:
        ...

    def __ne__(self, other: Object) -> bool:
        ...
