"""GDB helper commands and pretty-printers for TileLang / TVM C++ AST debugging.

Loaded automatically by `.vscode/launch.json` via `-exec source .vscode/tvm_gdb.py`.

Usage in VS Code Debug Console (when stopped at a C++ breakpoint):
    -exec ptvm op
    -exec ptvm buffer
    -exec ptvm collector.infer_list_stmt_[0]
    -exec ptree f
"""

import re

import gdb


class PrintTVMCommand(gdb.Command):
    """Print a TVM ObjectRef through GDB output, not inferior stderr."""

    def __init__(self):
        super().__init__("ptvm", gdb.COMMAND_DATA, gdb.COMPLETE_EXPRESSION)

    def invoke(self, arg: str, from_tty: bool) -> None:
        expr = arg.strip()
        if not expr:
            gdb.write("Usage: ptvm <ObjectRef_expr>\n")
            return
        try:
            # Dump writes to the inferior's stderr, which cppdbg may route to
            # a separate terminal. Return the text and emit it via GDB instead.
            value = gdb.parse_and_eval(f"tvm::RedirectedReprPrinterMethod({expr})")
            # This workspace uses GCC/libstdc++ with the C++11 string ABI.
            length = int(value["_M_string_length"])
            output = value["_M_dataplus"]["_M_p"].string(length=length)
            gdb.write(output + "\n")
        except gdb.error as exc:
            gdb.write(f"[ptvm] printing {expr} failed: {exc}\n")


class TVMTree:
    """Read actual IR children from inferior memory (not parsed TVMScript).

    Targets this workspace's TVM-FFI ABI. Metadata, buffer definitions and
    type annotations are omitted to keep the statement/expression tree clear.
    """

    fields = ('body', 'block', 'seq', 'init', 'condition', 'then_case',
              'else_case', 'min', 'extent', 'value', 'a', 'b', 'args',
              'indices', 'predicate', 'true_value', 'false_value',
              'base', 'stride', 'vectors', 'source')

    def __init__(self):
        self.types = {}
        self.lines = []
        self.count = 0

    def pointer(self, value):
        typ = value.type.strip_typedefs()
        if typ.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
            value = value.referenced_value()
            typ = value.type.strip_typedefs()
        if typ.code == gdb.TYPE_CODE_PTR:
            return value
        if str(typ).endswith('Node'):
            return value.address
        return value['data_']['data_']

    def describe(self, ptr):
        raw = ptr.cast(gdb.lookup_type('TVMFFIObject').pointer()).dereference()
        index = int(raw['type_index'])
        if index not in self.types:
            info = gdb.parse_and_eval(f'(const TVMFFITypeInfo*)TVMFFIGetTypeInfo({index})').dereference()
            key = info['type_key']['data'].string(length=int(info['type_key']['size']))
            typ = None
            name = key.rsplit('.', 1)[-1]
            candidates = ['tvm::ffi::ArrayObj'] if key == 'ffi.Array' else [
                f'tvm::tirx::{name}Node', f'tvm::{name}Node']
            for candidate in candidates:
                try:
                    typ = gdb.lookup_type(candidate)
                    break
                except gdb.error:
                    pass
            self.types[index] = (key, typ)
        key, typ = self.types[index]
        return key, ptr.cast(typ.pointer()).dereference() if typ else None

    def text(self, value):
        cell = value['data_']['data_']
        if int(cell['type_index']) < 64:
            return cell['v_bytes'].address.cast(gdb.lookup_type('char').pointer()).string(
                length=int(cell['small_str_len']))
        address = cell['v_obj'].cast(gdb.lookup_type('char').pointer())
        address += gdb.lookup_type('TVMFFIObject').sizeof
        obj = address.cast(gdb.lookup_type('TVMFFIByteArray').pointer()).dereference()
        return obj['data'].string(length=int(obj['size']))

    def summary(self, name, node):
        if name in ('SBlock',):
            return self.text(node['name_hint'])
        if name in ('Var', 'SizeVar', 'Buffer', 'Op'):
            return self.text(node['name_hint'] if name in ('Var', 'SizeVar') else node['name'])
        field = {'For': 'loop_var', 'Call': 'op',
                 'BufferLoad': 'buffer', 'BufferStore': 'buffer'}.get(name)
        if field:
            key, child = self.describe(self.pointer(node[field]))
            return self.summary(key.rsplit('.', 1)[-1], child) if child is not None else key
        if name == 'AttrStmt':
            return self.text(node['attr_key'])
        return ''

    def visit(self, value, edge='', prefix='', last=True, depth=0, ancestors=()):
        if self.count >= 1200:
            return
        ptr = self.pointer(value)
        if not int(ptr):
            return
        key, node = self.describe(ptr)
        name = key.rsplit('.', 1)[-1]
        label = (edge + ': ' if edge else '') + name
        if name in ('IntImm', 'FloatImm') and node is not None:
            label += ' ' + str(node['value'])
        if node is not None:
            try:
                detail = self.summary(name, node)
                if detail:
                    label += ' ' + detail
            except (gdb.error, KeyError, TypeError):
                pass
        self.lines.append(prefix + (('└─ ' if last else '├─ ') if depth else '') + label)
        self.count += 1
        if int(ptr) in ancestors:
            self.lines[-1] += ' [cycle]'
            return
        if depth >= 40:
            self.lines[-1] += ' [depth limit]'
            return
        children = []
        if key == 'ffi.Array' and node is not None:
            size = int(node['size'])
            items = node['data'].cast(gdb.lookup_type('TVMFFIAny').pointer())
            for i in range(min(size, 1200)):
                item = items[i]
                # TVM-FFI object type indices start at 64.
                if int(item['type_index']) >= 64 and int(item['v_obj']):
                    children.append((f'[{i}]', item['v_obj']))
        elif node is not None and name not in ('Var', 'SizeVar', 'Buffer', 'Op'):
            for field in self.fields:
                try:
                    child = node[field]
                    child_ptr = self.pointer(child)
                    if int(child_ptr):
                        children.append((field, child_ptr))
                except (gdb.error, KeyError, TypeError):
                    pass
        child_prefix = prefix + (('   ' if last else '│  ') if depth else '')
        for i, (field, child) in enumerate(children):
            self.visit(child, field, child_prefix, i == len(children)-1,
                       depth+1, ancestors+(int(ptr),))


class PrintTVMTreeCommand(gdb.Command):
    """ptree EXPR: print the actual statement/expression node tree."""

    def __init__(self):
        super().__init__('ptree', gdb.COMMAND_DATA, gdb.COMPLETE_EXPRESSION)

    def invoke(self, arg, from_tty):
        if not arg.strip():
            gdb.write('Usage: ptree <ObjectRef_or_node_pointer>\n')
            return
        tree = TVMTree()
        try:
            try:
                value = gdb.parse_and_eval(arg)
            except gdb.error:
                # libstdc++ vector operator[] may be inlined out.
                match = re.fullmatch(r"(.+)\[(\d+)\]", arg.strip())
                if not match:
                    raise
                vector = gdb.parse_and_eval(match[1])
                start = vector['_M_impl']['_M_start']
                end = vector['_M_impl']['_M_finish']
                index = int(match[2])
                if not 0 <= index < int(end - start):
                    raise gdb.GdbError('vector index out of bounds')
                value = start[index]
            tree.visit(value)
            gdb.write('\n'.join(tree.lines) + '\n')
            if tree.count >= 1200:
                gdb.write('[ptree] truncated at 1200 nodes\n')
        except (gdb.error, KeyError, TypeError) as exc:
            gdb.write('\n'.join(tree.lines) + f'\n[ptree] {exc}\n')


class TVMVariableResolver(TVMTree):
    """Resolve runtime types by reading cached type indices; never run inferior code."""

    node_names = '''PrimFunc Var SizeVar Buffer BufferRegion MatchBufferRegion
        SBlock SBlockRealize SeqStmt For AttrStmt Evaluate BufferStore BufferLoad
        Call Cast Add Sub Mul Div Mod FloorDiv FloorMod Min Max EQ NE LT LE GT GE
        And Or Not Select Ramp Broadcast Shuffle IntImm FloatImm StringImm
        IfThenElse While LetStmt Let Bind Allocate AllocateConst DeclBuffer
        AssertStmt ProducerStore ProducerLoad Prefetch IterVar Op GlobalVar
        Layout Fragment PartialFragment'''.split()

    def __init__(self):
        super().__init__()
        self.loaded = False

    def describe(self, ptr):
        if not self.loaded:
            for name in self.node_names:
                for ns in ('tvm::tirx', 'tvm', 'tvm::tl'):
                    cpp = f'{ns}::{name}Node'
                    try:
                        typ = gdb.lookup_type(cpp)
                        symbol = f'{cpp}::_GetOrAllocRuntimeTypeIndex()::tindex'
                        index = int(gdb.parse_and_eval(f"(int)'{symbol}'"))
                        if index > 64:
                            self.types[index] = (name, typ)
                            break
                    except gdb.error:
                        pass
            try:
                typ = gdb.lookup_type('tvm::ffi::ArrayObj')
                self.types[71] = ('ffi.Array', typ)
            except gdb.error:
                pass
            self.loaded = True
        raw = ptr.cast(gdb.lookup_type('TVMFFIObject').pointer()).dereference()
        key, typ = self.types.get(int(raw['type_index']), ('TVMObject', None))
        return key, ptr.cast(typ.pointer()).dereference() if typ else None

_tvm_variable_resolver = TVMVariableResolver()


class TVMVariablePrinter:
    def __init__(self, key, node):
        self.key, self.node = key, node

    def to_string(self):
        name = self.key.rsplit('.', 1)[-1]
        if self.key == 'ffi.Array':
            return f'Array ({int(self.node["size"])} items)'
        if name in ('IntImm', 'FloatImm'):
            return f'{name} {self.node["value"]}'
        try:
            detail = _tvm_variable_resolver.summary(name, self.node)
        except (gdb.error, KeyError, TypeError):
            detail = ''
        return name + (' ' + detail if detail else '')

    def children(self):
        if self.key == 'ffi.Array':
            items = self.node['data'].cast(gdb.lookup_type('TVMFFIAny').pointer())
            for i in range(min(int(self.node['size']), 2000)):
                item = items[i]
                if int(item['type_index']) >= 64 and int(item['v_obj']):
                    _, node = _tvm_variable_resolver.describe(item['v_obj'])
                    yield f'[{i}]', node if node is not None else item
                else:
                    yield f'[{i}]', item
            return
        # Derived node fields expose IR children, without the ObjectRef/base chain.
        for field in self.node.type.strip_typedefs().fields():
            if field.is_base_class or not hasattr(field, 'bitpos') or not field.name:
                continue
            if field.name.startswith('_') or field.name == 'span':
                continue
            yield field.name, self.node[field.name]


class TVMStringPrinter:
    def __init__(self, value):
        self.value = value

    def to_string(self):
        return _tvm_variable_resolver.text(self.value)

    def display_hint(self):
        return 'string'


class TVMVariableLookup:
    name = 'tilelang-ir-tree'
    enabled = True

    def __call__(self, value):
        try:
            if value.is_optimized_out:
                return None
            if not str(value.type.strip_typedefs()).startswith('tvm::'):
                return None
            if str(value.type.strip_typedefs()) == 'tvm::ffi::String':
                return TVMStringPrinter(value)
            ptr = _tvm_variable_resolver.pointer(value)
            if not int(ptr):
                return None
            key, node = _tvm_variable_resolver.describe(ptr)
            if node is not None:
                return TVMVariablePrinter(key, node)
        except (gdb.error, KeyError, TypeError, ValueError):
            pass
        return None


# Reloading the helper replaces the old printer, so the current session can use it.
PrintTVMCommand()
PrintTVMTreeCommand()
gdb.pretty_printers[:] = [p for p in gdb.pretty_printers
                         if getattr(p, 'name', '') != 'tilelang-ir-tree']
gdb.pretty_printers.insert(0, TVMVariableLookup())
gdb.write('[tvm_gdb] Loaded ptvm, ptree and IR pretty-printers for Variables and Watch.\n')
