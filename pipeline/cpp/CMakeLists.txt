cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
project(SpatialMapping)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(DOpen3D_DIR=${OPEN3D_DIR})

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2 
)

FetchContent_MakeAvailable(nlohmann_json)

find_package(Open3D REQUIRED)

add_subdirectory(test)

add_executable(main main.cpp Integration.cpp)

target_link_libraries(main PRIVATE Open3D::Open3D)

