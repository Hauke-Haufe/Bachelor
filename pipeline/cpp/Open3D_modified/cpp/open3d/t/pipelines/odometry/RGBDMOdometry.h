// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file RGBDOdometry.h
/// All the 4x4 transformation in this file, from params to returns, are
/// Float64. Only convert to Float32 in kernel calls.

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/geometry/RGBDMImage.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {


OdometryResult RGBDMOdometryMultiScale(
        const t::geometry::RGBDMImage& source,
        const t::geometry::RGBDMImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        const float depth_scale = 1000.0f,
        const float depth_max = 3.0f,
        const std::vector<OdometryConvergenceCriteria>& criteria_list = {10, 5, 3},
        const Method method = Method::PointToPlane,
        const OdometryLossParams& params = OdometryLossParams());

OdometryResult RGBDMOdometryMultiScale(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDMImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        const float depth_scale = 1000.0f,
        const float depth_max = 3.0f,
        const std::vector<OdometryConvergenceCriteria>& criteria_list = {10, 5, 3},
        const Method method = Method::PointToPlane,
        const OdometryLossParams& params = OdometryLossParams());

OdometryResult RGBDMOdometryMultiScale(
        const t::geometry::RGBDMImage& source,
        const t::geometry::RGBDImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        const float depth_scale = 1000.0f,
        const float depth_max = 3.0f,
        const std::vector<OdometryConvergenceCriteria>& criteria_list = {10, 5, 3},
        const Method method = Method::PointToPlane,
        const OdometryLossParams& params = OdometryLossParams());

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
