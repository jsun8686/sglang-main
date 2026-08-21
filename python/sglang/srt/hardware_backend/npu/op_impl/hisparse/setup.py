#!/usr/bin/env python3
"""
setup.py — CANN 9.0.0 / Atlas A3 build script for HiSparse NPU kernels.

Builds:
  - AscendC kernels (ccec) via CMake ascendc_library
  - Host/pybind11 shared library (g++) via CMake add_library

Environment variables:
  ASCEND_HOME     CANN installation root (default /usr/local/Ascend/ascend-toolkit/latest)
  SOC_VERSION     SoC model (default ascend910_9391)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CANN_ROOT = os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
SOC_VERSION = os.environ.get("SOC_VERSION", "ascend910_9391")


class CMakeBuildExt(build_ext):
    def build_extension(self, ext: Extension):
        src_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        cmake_build = Path(self.build_temp).resolve() / "cmake_build"

        if cmake_build.exists():
            shutil.rmtree(cmake_build)
        cmake_build.mkdir(parents=True, exist_ok=True)

        cmake = shutil.which("cmake")
        if not cmake:
            raise FileNotFoundError("cmake not found, please install cmake >= 3.16")

        env = os.environ.copy()
        env["ASCEND_HOME"] = CANN_ROOT

        configure_cmd = [
            cmake, str(src_dir),
            f"-DASCEND_CANN_PACKAGE_PATH={CANN_ROOT}",
            f"-DSOC_VERSION={SOC_VERSION}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        print(f"[cmake] configure: SOC_VERSION={SOC_VERSION}, CANN={CANN_ROOT}")
        subprocess.check_call(configure_cmd, cwd=str(cmake_build), env=env)

        build_cmd = [cmake, "--build", ".", "-j", str(os.cpu_count() or 4)]
        print("[cmake] build ...")
        subprocess.check_call(build_cmd, cwd=str(cmake_build), env=env)

        candidates = list(cmake_build.rglob("hisparse_lru.so"))
        # CMakeLists sets LIBRARY_OUTPUT_DIRECTORY to the source directory,
        # so the freshly linked .so may land next to CMakeLists.txt instead
        # of under the build tree.
        src_so = src_dir / "hisparse_lru.so"
        if src_so.exists():
            candidates.append(src_so)
        if not candidates:
            raise FileNotFoundError(
                f"CMake did not produce hisparse_lru.so, check build logs under {cmake_build}"
            )
        built_so = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[cmake] built: {built_so}")

        out_lib = Path(self.build_lib) / self.get_ext_filename(ext.name)
        out_lib.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_so, out_lib)

        if self.inplace:
            inplace_path = Path(src_dir) / self.get_ext_filename(ext.name)
            shutil.copy2(built_so, inplace_path)
            print(f"\nBuild succeeded: {inplace_path}")
        else:
            print(f"\nBuild succeeded: {out_lib}")


setup(
    name="hisparse_lru",
    version="0.1.0",
    ext_modules=[Extension("hisparse_lru", sources=[])],
    cmdclass={"build_ext": CMakeBuildExt},
)
