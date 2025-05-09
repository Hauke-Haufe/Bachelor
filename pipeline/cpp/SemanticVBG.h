#pragma once
#include "SemanticVBG.h"
#include "custom_kernel/custom_integration.h"
#include <open3d/Open3D.h>

using namespace open3d;

class SemanticVBG{

    private:

        t::geometry::VoxelBlockGrid vgb_;
        int block_resolution_;
        float voxel_size_;
        std::unordered_map<std::string, int> name_attr_map_;

    public:

        SemanticVBG(float voxel_size,
            int64_t block_resolution,
            int64_t block_count,
            const core::Device &device);
        
        core::Tensor GetUniqueBlockCoordinates(
                const t::geometry::Image &depth,
                const core::Tensor &intrinsics,
                const core::Tensor &extrinsics,
                float depth_scale = 1000.0f,
                float depth_max = 3.0f,
                float trunc_voxel_multiplier = 8.0);
        
        void Integrate(const core::Tensor &block_coords,
                    const t::geometry::Image &depth,
                    const t::geometry::Image &color,
                    const core::Tensor &label,
                    const core::Tensor &instrinsics,
                    const core::Tensor &extrinsics,
                    float depth_scale = 1000.0f,
                    float depth_max = 3.0f,
                    float trunc_voxel_multiplier = 8.0f);

        static t::geometry::TensorMap ConstructTensorMap(const core::HashMap &block_hashmap,
            std::unordered_map<std::string, int> name_attr_map);

};