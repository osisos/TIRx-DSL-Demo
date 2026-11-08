"""Static interface for APIs exposed dynamically by the runtime."""
from typing import Any, Iterator
from tvm_ffi import register_object as _register_object
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import Buffer, Var

@_register_object('script.ir_builder.tirx.TIRFrame')
class TIRFrame(IRBuilderFrame):
    ...

@_register_object('script.ir_builder.tirx.PrimFuncFrame')
class PrimFuncFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.SSBlockFrame')
class SBlockFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.SBlockInitFrame')
class BlockInitFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.ForFrame')
class ForFrame(TIRFrame):
    # TVMScript parses for-loops instead of invoking Python iteration.
    def __iter__(self) -> Iterator[Any]: ...

    def __enter__(self) -> Var | list[Var]:
        ...

@_register_object('script.ir_builder.tirx.AssertFrame')
class AssertFrame(TIRFrame):
    ...

class LetFrame(TIRFrame):

    def __enter__(self) -> Var:
        ...

class AllocateFrame(TIRFrame):

    def __enter__(self) -> Buffer:
        ...

@_register_object('script.ir_builder.tirx.AttrFrame')
class AttrFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.WhileFrame')
class WhileFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.IfFrame')
class IfFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.ThenFrame')
class ThenFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.ElseFrame')
class ElseFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.DeclBufferFrame')
class DeclBufferFrame(TIRFrame):

    def __enter__(self) -> Buffer:
        ...

@_register_object('script.ir_builder.tirx.LaunchThreadFrame')
class LaunchThreadFrame(TIRFrame):

    def __enter__(self) -> Var:
        ...

@_register_object('script.ir_builder.tirx.ComposeOpFrame')
class ComposeOpFrame(TIRFrame):
    ...

@_register_object('script.ir_builder.tirx.AllocBufferFrame')
class AllocBufferFrame(TIRFrame):

    def __enter__(self) -> Buffer:
        ...

@_register_object('script.ir_builder.tirx.HintFrame')
class HintFrame(TIRFrame):
    ...
