# Exit on error
$ErrorActionPreference = "Stop"

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Delete build cache
Remove-Item -Recurse -Force "$ScriptDir\build" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$ScriptDir\build" | Out-Null

# Create install directory
New-Item -ItemType Directory -Force -Path "$ScriptDir\install" | Out-Null

# Compile Open3D
Set-Location "$ScriptDir\build"

cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_CUDA_MODULE=ON -DCMAKE_INSTALL_PREFIX="$ScriptDir/install" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --parallel --target INSTALL
cmake --install .