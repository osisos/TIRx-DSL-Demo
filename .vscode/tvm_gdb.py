"""GDB helper commands and pretty-printers for TileLang / TVM C++ AST debugging.

Loaded automatically by `.vscode/launch.json` via `-exec source .vscode/tvm_gdb.py`.

Usage in VS Code Debug Console (when stopped at a C++ breakpoint):
    -exec ptvm op
    -exec ptvm buffer
    -exec ptvm infer_list_
"""

import gdb


class PrintTVMCommand(gdb.Command):
    """Print any TVM / TileLang ObjectRef or Object* using tvm::Dump()."""

    def __init__(self):
        super().__init__("ptvm", gdb.COMMAND_DATA, gdb.COMPLETE_EXPRESSION)

    def invoke(self, arg: str, from_tty: bool) -> None:
        expr = arg.strip()
        if not expr:
            gdb.write("Usage: ptvm <ObjectRef_or_Object_ptr_expr>\n")
            return
        try:
            gdb.execute(f"call (void)tvm::Dump({expr})")
        except gdb.error as exc:
            gdb.write(f"[ptvm] tvm::Dump({expr}) failed: {exc}\n")


PrintTVMCommand()
gdb.write("[tvm_gdb] Loaded TileLang/TVM GDB helpers. Use `-exec ptvm <var>` in Debug Console to dump any TIR/TileLang node.\n")
