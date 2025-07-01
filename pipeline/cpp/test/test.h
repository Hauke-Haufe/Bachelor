#pragma once

#include <filesystem>
#include <open3d/core/Tensor.h>

void test_masked_odometry(std::filesystem::path run_path, open3d::core::Tensor intrinsic_matrix);