#include "SemanticVBG.h"
#include <open3d/Open3D.h>
#include <open3d/t/geometry/kernel/VoxelBlockGrid.h>


using namespace open3d;

SemanticVBG::SemanticVBG(
    float voxel_size,
    int64_t block_resolution,
    int64_t block_count, open3d::core::Device &device){
 
    
    std::vector<std::string> attr_names = {"tsdf", "weight", "color", "label"};
    std::vector<core::Dtype> attr_types = {core::Dtype::Float32,core::Dtype::UInt16, core::Dtype::UInt16, core::Dtype::UInt8};
    std::vector<core::SizeVector> attr_channels = {{1}, {1}, {3}, {1}};

    auto vbg_ =t::geometry::VoxelBlockGrid(
        attr_names, 
        attr_types,
        attr_channels,
        voxel_size,
        block_resolution,
        block_count,
        device);
}

core::Tensor SemanticVBG::GetUniqueBlockCoordinates(
    const t::geometry::Image &depth,
    const core::Tensor &intrinsics,
    const core::Tensor &extrinsics,
    float depth_scale ,
    float depth_max ,
    float trunc_voxel_multiplier ){
    
    return vbg_.GetUniqueBlockCoordinates(depth, intrinsics, extrinsics, depth_scale, depth_max, trunc_voxel_multiplier);
};

void SemanticVBG::Integrate(const core::Tensor &block_coords,
    const t::geometry::Image &depth,
    const t::geometry::Image &color,
    const core::Tensor &label,
    const core::Tensor &instrinsic,
    const core::Tensor &extrinsic,
    float depth_scale ,
    float depth_max ,
    float trunc_voxel_multiplier) {
    
    vbg_.SemanticIntegrate(block_coords, depth, color, label, instrinsic, extrinsic, depth_scale, depth_max, trunc_voxel_multiplier);
};
        
std::tuple<t::geometry::PointCloud, core::Tensor>  SemanticVBG::ExtractSemanticPointCloud(float weight_threshold,
                int estimated_point_number){
    return vbg_.ExtractSemanticPointCloud(weight_threshold, estimated_point_number);   
};
