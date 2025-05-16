
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$OPEN3D_DIR = Join-Path $SCRIPT_DIR "Open3D_modified\install\CMake"
$BUILD_DIR = Join-Path $SCRIPT_DIR "build"

# Create the build directory if it doesn't exist
if (-Not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
}

# Change to the build directory
Set-Location $BUILD_DIR



# Run CMake with Debug configuration and Open3D directory
cmake -DCMAKE_BUILD_TYPE=Release -DOpen3D_DIR="$OPEN3D_DIR" ..

cmake --build . --config Release
