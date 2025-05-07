#pragma once
#include <open3d/Open3D.h>

using namespace open3d;

class SemanticVBG{

    private:

        t::geometry::VoxelBlockGrid vgb_;

    public:

        SemanticVBG();
        
        core::Tensor GetUniqueBlockCoordinated(
                const t::geometry::Image &depth,
                const core::Tensor &intrinsics,
                const core::Tensor &extrinsics,
                float depth_scale,
                float depth_max,
                float trunc_voxel_multiplier);
        
        void Integrate(const core::Tensor &block_coords,
                    const t::geometry::Image &depth,
                    const t::geometry::Image &color,
                    const core::Tensor &label,
                    const core::Tensor &instrinsics,
                    const core::Tensor &extrinsics,
                    float depth_scale = 1000.0f,
                    float depth_max = 3.0f,
                    float trunc_voxel_multiplier = 8.0f);
    


};