#pragma once
#include <open3d/Open3D.h>

using index_t = int;

void custom_Integrate(const open3d::core::Tensor& depth,
    const open3d::core::Tensor& color,
    const open3d::core::Tensor& label,
    const open3d::core::Tensor& block_indices,
    const open3d::core::Tensor& block_keys,
    open3d::t::geometry::TensorMap& block_value_map,
    const open3d::core::Tensor& depth_intrinsic,
    const open3d::core::Tensor& color_intrinsic,
    const open3d::core::Tensor& extrinsic,
    index_t resolution,
    float voxel_size,
    float sdf_trunc,
    float depth_scale,
    float depth_max);

template <typename input_depth_t,typename input_color_t,typename input_label_t,typename tsdf_t,typename weight_t,typename color_t, typename label_t>
void CustomIntegrateCPU(const open3d::core::Tensor& depth,
         const open3d::core::Tensor& color,
         const open3d::core::Tensor& label,
         const open3d::core::Tensor& indices,
         const open3d::core::Tensor& block_keys,
         open3d::t::geometry::TensorMap& block_value_map,
         const open3d::core::Tensor& depth_intrinsic,
         const open3d::core::Tensor& color_intrinsic,
         const open3d::core::Tensor& extrinsics,
         index_t resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_max);

template <typename input_depth_t,typename input_color_t,typename input_label_t,typename tsdf_t,typename weight_t,typename color_t, typename label_t>
void CustomIntegrateCUDA(const open3d::core::Tensor& depth,
         const open3d::core::Tensor& color,
         const open3d::core::Tensor& label,
         const open3d::core::Tensor& indices,
         const open3d::core::Tensor& block_keys,
         open3d::t::geometry::TensorMap& block_value_map,
         const open3d::core::Tensor& depth_intrinsic,
         const open3d::core::Tensor& color_intrinsic,
         const open3d::core::Tensor& extrinsics,
         index_t resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_max);