#pragma once

#include <filesystem>
#include <open3d/core/Tensor.h>
#include "open3d/t/pipelines/odometry/RGBDMOdometry.h"
#include "test_dataset.h"

enum class SlamMethod{
    Raw, 
    Masked, 
    WeightMask
};

void test_odometry(
    SubDataset dataset, 
    open3d::t::pipelines::odometry::Method method, 
    open3d::t::pipelines::odometry::MaskMethod m_mehtod,
    open3d::core::Device device, bool normal);

void test_odometry_optimize(SubDataset data,
                    t::pipelines::odometry::Method method, 
                    t::pipelines::odometry::MaskMethod m_mehtod,
                    core::Device  device,
                    bool normal);
void test_slam(
    SubDataset dataset,
    open3d::core::Device  device,
    SlamMethod mehtod);

void test_slam_robust(SubDataset dataset,
    open3d::core::Device  device,
    SlamMethod mehtod);

void test_convergence_speed_time(SubDataset data,
                    t::pipelines::odometry::Method method, 
                    t::pipelines::odometry::MaskMethod m_mehtod,
                    core::Device  device,
                    bool normal);

