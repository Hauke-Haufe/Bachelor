cmake_minimum_required(VERSION 3.22)

project(SpatilalMapping)

set(CMAKE_PREFIX_PATH "/home/nb-messen-07/open3d_install/lib/cmake/Open3D")
find_package(Open3D REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Open3D::Open3D)

