cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
project(SpatilalMapping)

#set(CMAKE_PREFIX_PATH "/home/nb-messen-07/open3d_install/lib/cmake/Open3D")
#set(CMAKE_PREFIX_PATH "C:/src/Open3d")
find_package(Open3D REQUIRED)

#include_directories(custom_kernel)

add_executable(main main.cpp 
    custom_kernel/custom_integration.cpp
    SemanticVBG.cpp)

target_link_libraries(main PRIVATE Open3D::Open3D)

