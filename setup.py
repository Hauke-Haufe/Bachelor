#!/usr/bin/env python3
import subprocess
import sys
import venv
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

def run(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd, shell=(os.name == "nt"))

def create_dirs():
    print("[1/5] Creating base directories...")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "image_segmentation" / "lib").mkdir(parents=True, exist_ok=True)

def setup_venv():
    print("[2/5] Setting up Python virtual environment...")
    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True).create(str(venv_dir))

    pip = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip"
    python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
   
    req = ROOT / "requirements.txt"
    if req.is_file():
        run([str(pip), "install", "-r", str(req)])
    else:
        print("WARNING: requirements.txt not found, skipping.")

def clone_deeplab():
    print("[3/5] Cloning DeepLabV3Plus-Pytorch...")
    lib_dir = ROOT / "image_segmentation" / "lib"
    deeplab_dir = lib_dir / "Deeplab"
    if not deeplab_dir.exists():
        run(["git", "clone", "https://github.com/VainF/DeepLabV3Plus-Pytorch.git", "tmp_deeplab"], cwd=lib_dir)
        (lib_dir / "tmp_deeplab").rename(deeplab_dir)
    else:
        print("DeepLab already exists, skipping.")

def clone_open3d():
    print("[4/5] Cloning and building Open3D fork...")
    fork_dir = ROOT / "odometry" / "lib" / "Open3D_fork"
    fork_dir.parent.mkdir(parents=True, exist_ok=True)

    if not fork_dir.exists():
        run(["git", "clone", "https://github.com/Hauke-Haufe/Open3D_fork.git", str(fork_dir)], cwd=fork_dir.parent)
    else:
        print("Open3D fork already exists, skipping clone.")

def create_data_placeholders():
    print("[5/5] Creating data placeholders...")
    dataset_dir = ROOT / "dataset"
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    odometry_dir = ROOT / "data" / "odometry"
    if not odometry_dir.exists():
        odometry_dir.mkdir(parents=True)
        (odometry_dir / "testResults").mkdir(parents=True) 
        (odometry_dir / "testConfigs").mkdir(parents=True)
        (odometry_dir/ "TUMDataset").mkdir(parents=True)

def main():
    create_dirs()
    setup_venv()
    clone_deeplab()
    clone_open3d()
    create_data_placeholders()

    print("\n=== Setup complete! ===")
    print("Activate the environment with:")
    if os.name == "nt":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")

if __name__ == "__main__":
    main()