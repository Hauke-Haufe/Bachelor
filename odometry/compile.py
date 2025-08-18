#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
CPP_DIR = ROOT / "odometry" / "cpp"   
BUILD_DIR = CPP_DIR / "build"
EXEC_DIR = ROOT / "executables"

def run(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd, shell=(os.name == "nt"))

def compile_project(open3d_dir: Path, build_type="Release"):
    """
    Configure and compile the CMake project.

    Args:
        open3d_dir (Path): Path to the Open3D CMake package (Open3D install/lib/cmake/Open3D).
        build_type (str): "Release" or "Debug".
    """

    BUILD_DIR.mkdir(exist_ok=True)
    run([
        "cmake",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DOpen3D_DIR={open3d_dir}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={EXEC_DIR}",
        ".."
    ], cwd=BUILD_DIR)

    run([
        "cmake",
        "--build", ".",
        "--parallel"
    ], cwd=BUILD_DIR)

    print(f"\n Build finished ({build_type}). Executables are in {EXEC_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OdometryTests with Open3D")
    parser.add_argument("--open3d_dir", required=True,
                        help="Path to Open3D cmake directory (e.g. /path/to/Open3D/install/lib/cmake/Open3D)")
    parser.add_argument("--build_type", default="Release", choices=["Release", "Debug"],
                        help="Build type (default: Release)")

    args = parser.parse_args()
    compile_project(Path(args.open3d_dir).resolve(), args.build_type)