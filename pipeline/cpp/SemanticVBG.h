#pragma once
#include <open3d/Open3D.h>


// wrapper class for a VoxelBlockGrid with a Integration that integrates semantic information
class SemanticVBG{

    private:

        open3d::t::geometry::VoxelBlockGrid vbg_;

    public:

        SemanticVBG(float voxel_size,
            int64_t block_resolution,int64_t block_count, 
            open3d::core::Device &device);
        
        open3d::core::Tensor GetUniqueBlockCoordinates(
                const open3d::t::geometry::Image &depth,
                const open3d::core::Tensor &intrinsics,
                const open3d::core::Tensor &extrinsics,
                float depth_scale = 1000.0f,
                float depth_max = 3.0f,
                float trunc_voxel_multiplier = 8.0);
        
        void Integrate(const open3d::core::Tensor &block_coords,
                    const open3d::t::geometry::Image &depth,
                    const open3d::t::geometry::Image &color,
                    const open3d::core::Tensor &label,
                    const open3d::core::Tensor &instrinsics,
                    const open3d::core::Tensor &extrinsics,
                    float depth_scale = 1000.0f,
                    float depth_max = 3.0f,
                    float trunc_voxel_multiplier = 8.0f);

};