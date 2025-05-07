#set(Python3_EXECUTABLE "path/to/python.exe" CACHE FILEPATH "")
#set(Python3_INCLUDE_DIR "path/to/include" CACHE PATH "")
#set(Python3_LIBRARY "path/to/python311.lib" CACHE FILEPATH "")
#set(Python3_FOUND TRUE CACHE BOOL "")
#set(Python3_VERSION "3.11.0" CACHE STRING "")
#set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "" FORCE)

param(
    [string]$path
)

cd $path

mkdir build
cd build

# this doesent build the realsense support
cmake -DBUILD_CUDA_MODULE=true  -DPython3_ROOT=C:\Users\Haufe\Desktop\beachlor\code\.venv\Scripts\python.exe  -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX="C:\src\Open3d" ..
cmake --build . --config Release --target INSTALL

