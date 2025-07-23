#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/IJsonConvertibleIO.h>
#include <open3d/t/pipelines/odometry/RGBDMOdometry.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>

#include "test/test.h"
#include "test/test_dataset.h"
#include <filesystem>

using namespace open3d;
namespace fs =  std::filesystem;

int main(){
    
    //open3d::utility::Logger::GetInstance().SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    core::Device device(core::Device::DeviceType::CUDA, 0);
    
    /*
    test_odometry(SubDataset::walking_static, t::pipelines::odometry::Method::PointToPlane, 
    t::pipelines::odometry::MaskMethod::CompleteMask, device, false);
    */
    
    test_slam(SubDataset::walking_static , device, SlamMethod::Masked);
    
    
    /*
    
    */
    /*
    

    t::geometry::Image source_color;
    t::geometry::Image target_color;
    t::geometry::Image source_depth;
    t::geometry::Image target_depth;
    t::geometry::Image source_mask; 
    t::geometry::Image target_mask; 

    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/color/image1744.png", source_color);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/depth/image1744.png", source_depth);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/color/image1747.png", target_color);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/depth/image1747.png", target_depth);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/masks/image1747.png", target_mask);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/masks/image1744.png", source_mask);

    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertibleFromJSON("/home/hauke/code/Beachlor/data/intrinsics/intrinsics.json", intrinsics);
    auto intrinsics_matrix = core::eigen_converter::EigenMatrixToTensor(intrinsics.intrinsic_matrix_);

    auto t = t::pipelines::odometry::ComputeResidualMap(
        t::geometry::RGBDImage(source_color, source_depth),
        t::geometry::RGBDImage(target_color, target_depth),
        core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0")),
        intrinsics_matrix, t::pipelines::odometry::Method::Hybrid, 
        1000.0f, 5.0f
    );

    t.AsTensor().Save("r_map.npy");

    auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(
        t::geometry::RGBDMImage(source_color, source_depth, source_mask), 
        t::geometry::RGBDMImage(target_color, target_depth, target_mask),
        intrinsics_matrix, core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
        1000.0f, 5.0f, {10, 10, 10}, t::pipelines::odometry::Method::Hybrid, 
        t::pipelines::odometry::OdometryLossParams()

    );

    auto t_new = t::pipelines::odometry::ComputeResidualMap(
        t::geometry::RGBDImage(source_color, source_depth),
        t::geometry::RGBDImage(target_color, target_depth),
        result.transformation_,
        intrinsics_matrix, t::pipelines::odometry::Method::Hybrid, 
        1000.0f, 5.0f
    );
    t_new.AsTensor().Save("r_mapt.npy");
    */

    /*auto t = core::Tensor::Load("/home/hauke/code/Beachlor/data/test_dataset/rgbd_dataset_freiburg3_walking_xyz/mask/1341846313.592026.png.npy");
    auto t_new = t.To(core::Dtype::Bool).To(core::Dtype::Float32);
    t::io::WriteNpy("t.npy", t::geometry::Image(t_new).FilterGaussian(45, 15).AsTensor());*/



    return 0;
};

