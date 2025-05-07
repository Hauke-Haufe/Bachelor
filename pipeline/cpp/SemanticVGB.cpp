#include "SemanticVGB.h"
#inlcude "custom_kernel/custom_integration.cpp"
#include <open3d/Open3D.h>


using namespace open3d;

SemanticVBG::SemanticVBG(){
    
    std::vector<std::string> atr_names = {"tsdf", "weight", "color", "label"};
    std::vector<core::Dtype> atr_types = {core::Dtype::Float32,core::Dtype::UInt16, core::Dtype::UInt16, core::Dtype::UInt8};
    std::vector<core::SizeVector> channels = {{1}, {1}, {3}, {1}};
    core::Device device("CUDA:0");

    t::geometry::VoxelBlockGrid vgb_ =t::geometry::VoxelBlockGrid(
        atr_names, 
        atr_types,
        channels,
        0.005,
        16,
        10000,
        device
    );

};

core::Tensor SemanticVBG::GetUniqueBlockCoordinated(
    const t::geometry::Image &depth,
    const core::Tensor &intrinsics,
    const core::Tensor &extrinsics,
    float depth_scale,
    float depth_max,
    float trunc_voxel_multiplier){
    
    return vgb_.GetUniqueBlockCoordinated(depth, intrinsics, extrinsics, depth_scale, depth_max, trunc_voxel_multiplier)
};

void SemanticVBG::Integrate(const core::Tensor &block_coords,
    const t::geometry::Image &depth,
    const t::geometry::Image &color,
    const core::Tensor &label,
    const core::Tensor &instrinsic,
    const core::Tensor &extrinsic,
    float depth_scale = 1000.0f,
    float depth_max = 3.0f,
    float trunc_voxel_multiplier = 8.0f) {
    

    bool integrate_color = color.AsTensor().NumElements() > 0;

    core::Tensor buf_indices, masks;
    vgb_.block_hashmap_->Activate(block_coords, buf_indices, masks);
    vgb_.block_hashmap_->Find(block_coords, buf_indices, masks);

    core::Tensor block_keys = vgb_.block_hashmap_->GetKeyTensor();
    TensorMap block_value_map =
            ConstructTensorMap(*block_hashmap_, name_attr_map_);

    customkernel::voxel_grid::Integrate(
            depth.AsTensor(), color.AsTensor(), buf_indices, block_keys,
            block_value_map, instrinsic, instrinsic, extrinsic,
            vgb_.block_resolution_, vgb_.voxel_size_,
            vgb_.voxel_size_ * trunc_voxel_multiplier, depth_scale, depth_max);
};
        

