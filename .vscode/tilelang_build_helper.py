#!/usr/bin/env python3
"""Build helper for TileLang C++ compiler development.

Modes:
  incremental   : Fast incremental build of `libtilelang.so` via Ninja (1~2s).
  debug-passes  : Patch `build.ninja` so TileLang passes (`src/transform/`, `src/op/`)
                  compile with `-O0 -g3 -UNDEBUG -fno-inline` and `-DTVM_LOG_DEBUG=1`
                  (while keeping `3rdparty/tvm` and `src/layout/` at `-O2` so layout
                  validation stays fast), then run Ninja.
  fast-rel      : Restore all `tilelang_objs` back to `-O2 -g -DNDEBUG` and build.
  reconfigure   : Run full CMake reconfiguration (`~/dev/scripts/build-tilelang.sh`).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TILELANG_ROOT = Path("/home/kael/dev/src/tilelang")
BUILD_DIR = TILELANG_ROOT / "build"
BUILD_NINJA = BUILD_DIR / "build.ninja"
BUILD_SCRIPT = Path("/home/kael/dev/scripts/build-tilelang.sh")
JOBS = os.environ.get("BUILD_JOBS", "6")


def _patch_tilelang_ninja(mode: str) -> int:
    """Patch only `CMakeFiles/tilelang_objs.dir` blocks in `build.ninja`."""
    if not BUILD_NINJA.is_file():
        print(f"[tilelang-build] Missing {BUILD_NINJA}, running full CMake configure first...")
        subprocess.run(["bash", str(BUILD_SCRIPT)], check=True)

    content = BUILD_NINJA.read_text(encoding="utf-8")
    blocks = content.split("\n\n")
    patched_count = 0

    for idx, block in enumerate(blocks):
        if "build CMakeFiles/tilelang_objs.dir/" not in block:
            continue

        is_layout_tu = "CMakeFiles/tilelang_objs.dir/src/layout/" in block

        if mode == "debug-passes":
            # Keep src/layout/ at -O2 so bounded exhaustive layout validation stays fast,
            # while enabling -O0 -g3 -fno-inline + DLOG on all transform/op/codegen TUs.
            new_flags = (
                "  FLAGS = -O2 -g3 -UNDEBUG -fPIC"
                if is_layout_tu
                else "  FLAGS = -O0 -g3 -UNDEBUG -fno-inline -fPIC"
            )
            updated = re.sub(r"^  FLAGS = .*$", new_flags, block, flags=re.MULTILINE)
            if "-DTVM_LOG_DEBUG" not in updated:
                updated = re.sub(
                    r"^(  DEFINES = .*)$",
                    r"\1 -DTVM_LOG_DEBUG=1",
                    updated,
                    flags=re.MULTILINE,
                )
        elif mode == "fast-rel":
            updated = re.sub(
                r"^  FLAGS = .*$",
                "  FLAGS = -O2 -g -DNDEBUG -fPIC",
                block,
                flags=re.MULTILINE,
            )
            updated = updated.replace(" -DTVM_LOG_DEBUG=1", "")
        else:
            continue

        if updated != block:
            blocks[idx] = updated
            patched_count += 1

    if patched_count > 0:
        BUILD_NINJA.write_text("\n\n".join(blocks), encoding="utf-8")
    return patched_count


def run_ninja() -> int:
    cmd = ["/usr/bin/ninja", "-C", str(BUILD_DIR), "-j", str(JOBS), "tilelang"]
    print(f"[tilelang-build] Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "incremental"
    if mode == "incremental":
        return run_ninja()
    if mode in ("debug-passes", "fast-rel"):
        changed = _patch_tilelang_ninja(mode)
        label = (
            "Debug Passes (-O0 -g3 -UNDEBUG + TVM_LOG_DEBUG on transform/op/codegen)"
            if mode == "debug-passes"
            else "RelWithDebInfo (-O2 -g -DNDEBUG)"
        )
        print(f"[tilelang-build] Switched {changed} TileLang TUs to: {label}")
        return run_ninja()
    if mode == "reconfigure":
        return subprocess.call(["bash", str(BUILD_SCRIPT)])

    print(f"Unknown mode: {mode!r}. Expected: incremental | debug-passes | fast-rel | reconfigure")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
