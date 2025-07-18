#pragma once

#include <filesystem>
#include <open3d/core/Tensor.h>
#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"

void test_masked_odometry(
    std::filesystem::path run_path, 
    open3d::t::pipelines::odometry::Method method, 
    open3d::t::pipelines::odometry::MaskMethod m_mehtod,
    open3d::core::Device device);