add_library(Testutil STATIC TestDataset.cpp TestUtil.cpp)
target_link_libraries(Testutil PRIVATE Open3D::Open3D)

add_executable(OdometryTest OdometryTest.cpp)
target_link_libraries(OdometryTest PRIVATE Open3D::Open3D nlohmann_json::nlohmann_json Testutil)

add_executable(SlamTest SlamTest.cpp)
target_link_libraries(SlamTest PRIVATE Open3D::Open3D nlohmann_json::nlohmann_json Testutil) 