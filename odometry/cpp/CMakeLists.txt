cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
project(OdometryTests)

set(DOpen3D_DIR=${OPEN3D_DIR} CACHE PATH "Path to Open3D installation")

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2 
)

FetchContent_MakeAvailable(nlohmann_json)

find_package(Open3D REQUIRED)

add_library(Testutil STATIC TestDataset.cpp TestUtil.cpp)
target_link_libraries(Testutil PRIVATE Open3D::Open3D)

add_executable(mismain MiselaniousMain.cpp)
target_link_libraries(mismain PRIVATE Open3D::Open3D)

add_executable(OdometryTest OdometryTest.cpp)
target_link_libraries(OdometryTest PRIVATE Open3D::Open3D nlohmann_json::nlohmann_json Testutil)

add_executable(SlamTest SlamTest.cpp)
target_link_libraries(SlamTest PRIVATE Open3D::Open3D nlohmann_json::nlohmann_json Testutil) 

add_executable(t quick_test.cpp)
target_link_libraries(t PRIVATE Open3D::Open3D) 
