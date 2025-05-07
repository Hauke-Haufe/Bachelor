#pragma once
#include <open3d/Open3D.h>

using namespace open3d;

class SemanticVBG{

    public:

        SemanticVGB();
        
        void Integrate(const core::Tensor &block_coords,
                    const t::geometry::Image &depth,
                    const t::geometry::Image &color,
                    const core::Tensor &instrinsics,
                    const core::Tensor &extrinsics,
                    float depth_scale = 1000.0f,
                    float depth_max = 3.0f,
                    float trunc_voxel_multiplier = 8.0f
                    );
    
    private:

        std::shared_ptr<t::geometry::VoxelBlockGrid> vgb;


    
}