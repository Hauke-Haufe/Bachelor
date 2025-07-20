#include "Integration.h"

using namespace open3d;
namespace fs = std::filesystem;

std::unique_ptr<t::geometry::VoxelBlockGrid> create_vgb(){

    std::vector<std::string> atr_names = {"tsdf", "weight", "color"};
    std::vector<core::Dtype> atr_types = {core::Dtype::Float32,core::Dtype::Float32, core::Dtype::Float32};
    std::vector<core::SizeVector> channels = {{1}, {1}, {3}};
    core::Device device("CPU:0");

    return std::make_unique<t::geometry::VoxelBlockGrid>(
        atr_names, 
        atr_types,
        channels,
        0.005,
        16,
        10000,
        device
    );

}

std::unique_ptr<t::geometry::VoxelBlockGrid> integrate(
    pipelines::registration::PoseGraph& Posegraph,
    const std::vector<fs::path>& color_images,
    const std::vector<fs::path>& depth_images, 
    core::Tensor& instrinsics, 
    float depth_scale,
    float depth_max
    ){
    
    std::unique_ptr<t::geometry::VoxelBlockGrid> vgb;
    vgb = create_vgb();

    t::geometry::Image color_image;
    t::geometry::Image depth_image;
    core::Device cuda_device(core::Device::DeviceType::CPU, 0);
    
    for(int i = 0; i < color_images.size()-1; i++){

        t::io::ReadImage(color_images[i], color_image);
        t::io::ReadImage(depth_images[i], depth_image);
        
        auto pose = Posegraph.nodes_[i].pose_;
        Eigen::Matrix4d inverse = pose.inverse();

        auto inverse_t = core::eigen_converter::EigenMatrixToTensor(inverse);

        auto frustum_block_coords = vgb->GetUniqueBlockCoordinates(
            depth_image.To(cuda_device),
            instrinsics,
            inverse_t, 
            depth_scale,
            depth_max
        );

        vgb->Integrate(
            frustum_block_coords,
            depth_image.To(cuda_device),
            color_image.To(cuda_device),
            instrinsics,
            inverse_t,
            depth_scale,
            depth_max
        );

    }

    return vgb;
}