
$BUILD_DIR = "build"


# Change to build directory
Set-Location $BUILD_DIR

# Run CMake with Debug configuration and Open3D directory
cmake .. -DCMAKE_BUILD_TYPE=Release 

cmake --build .
