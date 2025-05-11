#include <open3d/core/CUDAUtils.h>
#include <open3d/core/ParallelFor.h>
#include <open3d/core/Dispatch.h>
#include <open3d/core/Dtype.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/core/SizeVector.h>
#include <open3d/core/Tensor.h>
#include <open3d/core/hashmap/Dispatch.h>
#include <open3d/t/geometry/Utility.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/t/geometry/kernel/GeometryMacros.h>
#include <open3d/t/geometry/kernel/VoxelBlockGrid.h>
#include <open3d/utility/Logging.h>
#include <open3d/utility/Timer.h>
#include "custom_integration.h"

using index_t = int;
using ArrayIndexer = open3d::t::geometry::kernel::TArrayIndexer<index_t>; 

template <typename input_depth_t,typename input_color_t,typename input_label_t,typename tsdf_t,typename weight_t,typename color_t, typename label_t>
#if defined(__CUDACC__)
void CustomIntegrateCUDA
#else
void CustomIntegrateCPU
#endif
        (const open3d::core::Tensor& depth,
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
         float depth_max) {
    // Parameters
    index_t resolution2 = resolution * resolution;
    index_t resolution3 = resolution2 * resolution;

    open3d::t::geometry::kernel::TransformIndexer transform_indexer(depth_intrinsic, extrinsics, voxel_size);
    open3d::t::geometry::kernel::TransformIndexer colormap_indexer(
            color_intrinsic,
            open3d::core::Tensor::Eye(4, open3d::core::Dtype::Float64, open3d::core::Device("CPU:0")));
    open3d::t::geometry::kernel::TransformIndexer labelmap_indexer(
            color_intrinsic,
            open3d::core::Tensor::Eye(4, open3d::core::Dtype::Float64, open3d::core::Device("CPU:0")));
    

    ArrayIndexer voxel_indexer({resolution, resolution, resolution});

    ArrayIndexer block_keys_indexer(block_keys, 1);
    ArrayIndexer depth_indexer(depth, 2);
    open3d::core::Device device = block_keys.GetDevice();

    const index_t* indices_ptr = indices.GetDataPtr<index_t>();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        open3d::utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    tsdf_t* tsdf_base_ptr = block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();

    bool integrate_color =
            block_value_map.Contains("color") && color.NumElements() > 0;
    color_t* color_base_ptr = nullptr;
    ArrayIndexer color_indexer;

    float color_multiplier = 1.0;
    if (integrate_color) {
        color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
        color_indexer = ArrayIndexer(color, 2);

        // Float32: [0, 1] -> [0, 255]
        if (color.GetDtype() == open3d::core::Float32) {
            color_multiplier = 255.0;
        }
    }

    bool integrate_label = block_value_map.Contains("label");
    label_t* label_base_ptr = nullptr;
    ArrayIndexer label_indexer;

    if (integrate_label){
        label_base_ptr = block_value_map.at("label").GetDataPtr<label_t>();
        label_indexer = ArrayIndexer(label, 2);
    }
    

    index_t n = indices.GetLength() * resolution3;
    open3d::core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t block_idx = indices_ptr[workload_idx / resolution3];
        index_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        index_t* block_key_ptr =
                block_keys_indexer.GetDataPtr<index_t>(block_idx);
        index_t xb = block_key_ptr[0];
        index_t yb = block_key_ptr[1];
        index_t zb = block_key_ptr[2];

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // coordinate in world (in voxel)
        index_t x = xb * resolution + xv;
        index_t y = yb * resolution + yv;
        index_t z = zb * resolution + zv;

        // coordinate in camera (in voxel -> in meter)
        float xc, yc, zc, u, v;
        transform_indexer.RigidTransform(static_cast<float>(x),
                                         static_cast<float>(y),
                                         static_cast<float>(z), &xc, &yc, &zc);

        // coordinate in image (in pixel)
        transform_indexer.Project(xc, yc, zc, &u, &v);
        if (!depth_indexer.InBoundary(u, v)) {
            return;
        }

        index_t ui = static_cast<index_t>(u);
        index_t vi = static_cast<index_t>(v);

        // Associate image workload and compute SDF and
        // TSDF.
        float depth =
                *depth_indexer.GetDataPtr<input_depth_t>(ui, vi) / depth_scale;

        float sdf = depth - zc;
        if (depth <= 0 || depth > depth_max || zc <= 0 || sdf < -sdf_trunc) {
            return;
        }
        sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
        sdf /= sdf_trunc;

        index_t linear_idx = block_idx * resolution3 + voxel_idx;

        tsdf_t* tsdf_ptr = tsdf_base_ptr + linear_idx;
        weight_t* weight_ptr = weight_base_ptr + linear_idx;

        float inv_wsum = 1.0f / (*weight_ptr + 1);
        float weight = *weight_ptr;
        *tsdf_ptr = (weight * (*tsdf_ptr) + sdf) * inv_wsum;

        if (integrate_color) {
            color_t* color_ptr = color_base_ptr + 3 * linear_idx;

            // Unproject ui, vi with depth_intrinsic, then project back with
            // color_intrinsic
            float x, y, z;
            transform_indexer.Unproject(ui, vi, 1.0, &x, &y, &z);

            float uf, vf;
            colormap_indexer.Project(x, y, z, &uf, &vf);
            if (color_indexer.InBoundary(uf, vf)) {
                ui = round(uf);
                vi = round(vf);

                input_color_t* input_color_ptr =
                        color_indexer.GetDataPtr<input_color_t>(ui, vi);

                for (index_t i = 0; i < 3; ++i) {
                    color_ptr[i] = (weight * color_ptr[i] +
                                    input_color_ptr[i] * color_multiplier) *
                                   inv_wsum;
                }
            }
        }

        if (integrate_label){
            label_t* label_ptr = label_base_ptr + linear_idx;

            float x, y, z;
            transform_indexer.Unproject(ui, vi, 1.0, &x, &y, &z);

            float uf, vf;
            labelmap_indexer.Project(x,y,z, &uf, &vf);
            if (label_indexer.InBoundary(uf, vf)){
                ui = round(uf);
                vi = round(vf);

                input_label_t* input_label_ptr = label_indexer.GetDataPtr<input_label_t>(ui, vi);
                label_ptr = input_label_ptr;
            }

        }

        *weight_ptr = weight + 1;
    });
};
