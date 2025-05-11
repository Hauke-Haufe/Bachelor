#include "SemanticVBG.h"
#include <open3d/Open3D.h>
#include <open3d/t/geometry/kernel/VoxelBlockGrid.h>


using namespace open3d;

t::geometry::TensorMap SemanticVBG::ConstructTensorMap(
    const core::HashMap &block_hashmap,
    std::unordered_map<std::string, int> name_attr_map) {
    t::geometry::TensorMap tensor_map("tsdf");
    for (auto &v : name_attr_map) {
        std::string name = v.first;
        int buf_idx = v.second;
        tensor_map[name] = block_hashmap.GetValueTensor(buf_idx);
    }
    return tensor_map;
}

SemanticVBG::SemanticVBG(
    float voxel_size,
    int64_t block_resolution,
    int64_t block_count, open3d::core::Device &device): voxel_size_(voxel_size), block_resolution_(block_resolution) {
 
    
    std::vector<std::string> attr_names = {"tsdf", "weight", "color", "label"};
    std::vector<core::Dtype> attr_types = {core::Dtype::Float32,core::Dtype::UInt16, core::Dtype::UInt16, core::Dtype::UInt8};
    std::vector<core::SizeVector> attr_channels = {{1}, {1}, {3}, {1}};

    // Specify block element shapes and attribute names.
    std::vector<core::SizeVector> attr_element_shapes;
    core::SizeVector block_shape{block_resolution, block_resolution,block_resolution};
    size_t n_attrs = attr_names.size();

    for (size_t i = 0; i < n_attrs; ++i) {
        // Construct element shapes.
        core::SizeVector attr_channel = attr_channels[i];
        core::SizeVector block_shape_copy = block_shape;
        block_shape_copy.insert(block_shape_copy.end(), attr_channel.begin(),
                                attr_channel.end());
        attr_element_shapes.emplace_back(block_shape_copy);

        // Used for easier accessing via attribute names.
        name_attr_map_[attr_names[i]] = i;
    }   


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
    
    return vgb_.GetUniqueBlockCoordinates(depth, intrinsics, extrinsics, depth_scale, depth_max, trunc_voxel_multiplier);
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
    

    bool integrate_color = color.AsTensor().NumElements() > 0;

    core::Tensor buf_indices, masks;
    auto block_hashmap = vgb_.GetHashMap();
    block_hashmap.Activate(block_coords, buf_indices, masks);
    block_hashmap.Find(block_coords, buf_indices, masks);
    
    core::Tensor block_keys = block_hashmap.GetKeyTensor();
    t::geometry::TensorMap block_value_map = ConstructTensorMap(block_hashmap, name_attr_map_);

    t::geometry::kernel::voxel_grid::custom_Integrate(
            depth.AsTensor(), color.AsTensor(), label, buf_indices, block_keys,
            block_value_map, instrinsic, instrinsic, extrinsic,
            block_resolution_, voxel_size_,
            voxel_size_ * trunc_voxel_multiplier, depth_scale, depth_max);
};
        

