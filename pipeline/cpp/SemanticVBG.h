#pragma once
#include "custom_kernel/custom_integration.h"
#include <open3d/Open3D.h>

class SemanticVBG{

    private:

        open3d::t::geometry::VoxelBlockGrid vgb_;
        int block_resolution_;
        float voxel_size_;
        std::unordered_map<std::string, int> name_attr_map_;

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

        static open3d::t::geometry::TensorMap ConstructTensorMap(const open3d::core::HashMap &block_hashmap,
            std::unordered_map<std::string, int> name_attr_map);

};