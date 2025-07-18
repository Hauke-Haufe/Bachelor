cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
project(SpatialMapping)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(DOpen3D_DIR=${OPEN3D_DIR})

find_package(Open3D REQUIRED)

add_executable(main main.cpp
    Integration.cpp
    test/test.cpp
    test/test_dataset.cpp)

target_link_libraries(main PRIVATE Open3D::Open3D)

