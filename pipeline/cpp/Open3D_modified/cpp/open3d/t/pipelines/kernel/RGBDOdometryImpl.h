// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.
#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeOdometryResultPointToPlaneCPU(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& source_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

void ComputeOdometryResultIntensityCPU(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float intensity_huber_delta);

//-----------------------------------------------------------------
void ComputeDMaskOdometryResultHybridCPU(const core::Tensor& source_depth,
                                    const core::Tensor& target_depth,
                                    const core::Tensor& source_intensity,
                                    const core::Tensor& target_intensity,
                                    const core::Tensor& target_depth_dx,
                                    const core::Tensor& target_depth_dy,
                                    const core::Tensor& target_intensity_dx,
                                    const core::Tensor& target_intensity_dy,
                                    const core::Tensor& source_vertex_map,
                                    const core::Tensor& source_mask,
                                    const core::Tensor& target_mask,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    core::Tensor& delta,
                                    float& inlier_residual,
                                    int& inlier_count,
                                    float depth_outlier_trunc,
                                    const float depth_huber_delta,
                                    const float intensity_huber_delta);

void ComputeOdometryResultHybridCPU(const core::Tensor& source_depth,
                                    const core::Tensor& target_depth,
                                    const core::Tensor& source_intensity,
                                    const core::Tensor& target_intensity,
                                    const core::Tensor& target_depth_dx,
                                    const core::Tensor& target_depth_dy,
                                    const core::Tensor& target_intensity_dx,
                                    const core::Tensor& target_intensity_dy,
                                    const core::Tensor& source_vertex_map,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    core::Tensor& delta,
                                    float& inlier_residual,
                                    int& inlier_count,
                                    float depth_outlier_trunc,
                                    const float depth_huber_delta,
                                    const float intensity_huber_delta);

void ComputeOdometryInformationMatrixCPU(const core::Tensor& source_depth,
                                         const core::Tensor& target_depth,
                                         const core::Tensor& intrinsic,
                                         const core::Tensor& source_to_target,
                                         const float depth_outlier_trunc,
                                         core::Tensor& information);

void ComputeResidualMapIntensityCPU(const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor target_depth,
        const core::Tensor& source_vertex_map, 
        core::Tensor& residuals, 
        const core::Tensor& source_to_target, 
        const core::Tensor& intrinsics, 
        const float depth_outlier_trunc);

void ComputeResidualMapPointToPlaneCUDA(const core::Tensor& source_vertex_map, 
        const core::Tensor& target_vertex_map, 
        const core::Tensor& target_normal_map,
        core::Tensor& residuals, 
        const core::Tensor& source_to_target, 
        const core::Tensor& intrinsics, 
        const float depth_outlier_trunc);

#ifdef BUILD_CUDA_MODULE

void ComputeOdometryResultPointToPlaneCUDA(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& source_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

void ComputeOdometryResultIntensityCUDA(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float intensity_huber_delta);

void ComputeMaskOdometryResultHybridCUDA(const core::Tensor& source_depth,
                                const core::Tensor& target_depth,
                                const core::Tensor& source_intensity,
                                const core::Tensor& target_intensity,
                                const core::Tensor& target_depth_dx,
                                const core::Tensor& target_depth_dy,
                                const core::Tensor& target_intensity_dx,
                                const core::Tensor& target_intensity_dy,
                                const core::Tensor& source_vertex_map,
                                const core::Tensor& source_mask,
                                const core::Tensor& target_mask,
                                const core::Tensor& intrinsics,
                                const core::Tensor& init_source_to_target,
                                core::Tensor& delta,
                                float& inlier_residual,
                                int& inlier_count,
                                const float depth_outlier_trunc,
                                const float depth_huber_delta,
                                const float intensity_huber_delta);

void ComputeOdometryResultHybridCUDA(const core::Tensor& source_depth,
                                     const core::Tensor& target_depth,
                                     const core::Tensor& source_intensity,
                                     const core::Tensor& target_intensity,
                                     const core::Tensor& target_depth_dx,
                                     const core::Tensor& target_depth_dy,
                                     const core::Tensor& target_intensity_dx,
                                     const core::Tensor& target_intensity_dy,
                                     const core::Tensor& source_vertex_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     core::Tensor& delta,
                                     float& inlier_residual,
                                     int& inlier_count,
                                     const float depth_outlier_trunc,
                                     const float depth_huber_delta,
                                     const float intensity_huber_delta);

void ComputeOdometryInformationMatrixCUDA(const core::Tensor& source_depth,
                                          const core::Tensor& target_depth,
                                          const core::Tensor& intrinsic,
                                          const core::Tensor& source_to_target,
                                          const float square_dist_thr,
                                          core::Tensor& information);

void ComputeResidualMapIntensityCUDA(const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor target_depth,
        const core::Tensor& source_vertex_map, 
        core::Tensor& residuals, 
        const core::Tensor& source_to_target, 
        const core::Tensor& intrinsics, 
        const float depth_outlier_trunc);

void ComputeResidualMapPointToPlaneCUDA(const core::Tensor& source_vertex_map, 
        const core::Tensor& target_vertex_map, 
        const core::Tensor& target_normal_map,
        core::Tensor& residuals, 
        const core::Tensor& source_to_target, 
        const core::Tensor& intrinsics, 
        const float depth_outlier_trunc);

#endif


using t::geometry::kernel::NDArrayIndexer;
using t::geometry::kernel::TransformIndexer;

#ifndef __CUDACC__
using std::abs;
using std::isnan;
using std::max;
#endif

#if defined(__CUDACC__)
__device__ inline bool ComputeIntensityResidual
#else
inline bool ComputeIntensityResidual
#endif

    (int x, 
    int y, 
    const float depth_outlier_trunc,
    const NDArrayIndexer& source_vertex_indexer,
    const NDArrayIndexer& source_intensity_indexer, 
    const NDArrayIndexer& target_intensity_indexer, 
    const NDArrayIndexer& target_depth_indexer,
    const TransformIndexer& trans,
    float& residual
    ) {
    
    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3], u_tf, v_tf;
    trans.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    trans.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u_tf, &v_tf);
    int u_t = int(roundf(u_tf));
    int v_t = int(roundf(v_tf));

    if (T_source_to_target_v[2] < 0 ||
        !target_intensity_indexer.InBoundary(u_t, v_t)) {
        return false;
    }
    float depth_t = *target_depth_indexer.GetDataPtr<float>(u_t, v_t);
    float diff_D = depth_t - T_source_to_target_v[2];
    if (isnan(depth_t) || abs(diff_D) > depth_outlier_trunc) {
        return false;
    }
    
    residual = abs( *source_intensity_indexer.GetDataPtr<float>(x,y) - 
                *target_intensity_indexer.GetDataPtr<float>(u_t, v_t));
    
    return true;
}

#if defined(__CUDACC__)
__device__ inline bool ComputePointToPlaneResidual
#else
inline bool ComputePointToPlaneResidual
#endif
    (int x, 
    int y, 
    const float depth_outlier_trunc,
    const NDArrayIndexer& source_vertex_indexer,
    const NDArrayIndexer& target_vertex_indexer, 
    const NDArrayIndexer& target_normal_indexer,  
    const TransformIndexer& trans,
    float& residual
    ){

    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3], u, v;
    trans.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    trans.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u, &v);
    u = roundf(u);
    v = roundf(v);

    if (T_source_to_target_v[2] < 0 ||
        !target_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int ui = static_cast<int>(u);
    int vi = static_cast<int>(v);
    float* target_v = target_vertex_indexer.GetDataPtr<float>(ui, vi);
    float* target_n = target_normal_indexer.GetDataPtr<float>(ui, vi);
    if (isnan(target_v[0]) || isnan(target_n[0])) {
        return false;
    }

    residual = abs((T_source_to_target_v[0] - target_v[0]) * target_n[0] +
        (T_source_to_target_v[1] - target_v[1]) * target_n[1] +
        (T_source_to_target_v[2] - target_v[2]) * target_n[2]);

}


}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
