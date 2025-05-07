#include "SemanticVGB.h"
#include <open3d/Open3D.h>

using namespace open3d;

SemanticVBG::SemanticVGB(){
    
    std::vector<std::string> atr_names = {"tsdf", "weight", "color", "label"};
    std::vector<core::Dtype> atr_types = {core::Dtype::Float32,core::Dtype::UInt16, core::Dtype::Uint16, core::Dtype::UInt8};
    std::vector<core::SizeVector> channels = {{1}, {1}, {3}, {1}};
    core::Device device("CUDA:0");

    self-> vgb = std::make_unique<t::geometry::VoxelBlockGrid>(
        atr_names, 
        atr_types,
        channels,
        0.005,
        16,
        10000,
        device
    );

    return
}

core::Tensor SemanticVGB::GetUniqueBlockCoordinated(
    const t::geometry::Image &depth,
    const core::Tensor &intrinsics,
    const core::Tensor &extrinsics,
    float depth_scale,
    float depth_max,
    float trinc_voxel_multiplier){
    
    return vgb_.GetUniqueBlockCoordinated(depth_image, instrinsics, extrinsics, depth_scale, depth_max, trunc_voxel_multiplier)
}

void SemanticVBG::Integrate(const core::Tensor &block_coords,
    const t::geometry::Image &depth,
    const t::geometry::Image &color,
    const core::Tensor &instrinsics,
    const core::Tensor &extrinsics,
    float depth_scale = 1000.0f,
    float depth_max = 3.0f,
    float trunc_voxel_multiplier = 8.0f) {
    


        
}
