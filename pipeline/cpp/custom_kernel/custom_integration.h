#pragma once
#include <open3d/Open3D.h>

using namespace open3d;

void custom_Integrate(const core::Tensor& depth,
    const core::Tensor& color,
    const core::Tensor& label,
    const core::Tensor& block_indices,
    const core::Tensor& block_keys,
    t::geometry::TensorMap& block_value_map,
    const core::Tensor& depth_intrinsic,
    const core::Tensor& color_intrinsic,
    const core::Tensor& extrinsic,
    index_t resolution,
    float voxel_size,
    float sdf_trunc,
    float depth_scale,
    float depth_max) 