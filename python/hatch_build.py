"""
Hatch build hook: bundle the compiled OpenMat.so into the wheel.

pyproject.toml declares `artifacts = ["openmat/OpenMat.so"]`, but the library is
built by CMake outside the Python tree, so it has to be copied in before hatch
collects files.  Set OPENMAT_LIB to override the location; otherwise the repo's
build/ directory is used.

Without a library the hook only warns: an sdist or a metadata-only build should
not require a CUDA toolchain, and openmat/_clib.py falls back to $OPENMAT_LIB
and to <repo>/build/OpenMat.so at import time.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LIB_NAME = "OpenMat.so"


def _locate_library(root: Path) -> Optional[Path]:
    env = os.environ.get("OPENMAT_LIB")
    if env:
        return Path(env).resolve()
    # python/ -> repo root -> build/OpenMat.so
    candidate = root.parent / "build" / LIB_NAME
    return candidate if candidate.exists() else None


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        root = Path(self.root)
        dest = root / "openmat" / LIB_NAME
        src = _locate_library(root)

        if src is None or not src.exists():
            self.app.display_warning(
                f"{LIB_NAME} not found — building without a bundled library. "
                f"Build it with ./compile.sh, or set OPENMAT_LIB."
            )
            return

        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
            self.app.display_info(f"bundled {src} -> {dest}")

        build_data["artifacts"].append(f"openmat/{LIB_NAME}")
        build_data["force_include"][str(dest)] = f"openmat/{LIB_NAME}"
