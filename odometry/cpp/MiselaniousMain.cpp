#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/IJsonConvertibleIO.h>
#include <open3d/t/pipelines/odometry/RGBDMOdometry.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>
#include <open3d/t/io/ImageIO.h>

#include <filesystem>

using namespace open3d;
namespace fs =  std::filesystem;

int main(){
    
    t::geometry::Image source_color;
    t::geometry::Image target_color;
    t::geometry::Image source_depth;
    t::geometry::Image target_depth;
    t::geometry::Image source_mask; 
    t::geometry::Image target_mask; 

    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/color/image1765.png", source_color);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/depth/image1765.png", source_depth);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/color/image1774.png", target_color);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/depth/image1774.png", target_depth);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/masks/image1774.png", target_mask);
    t::io::ReadImageFromPNG("/home/hauke/code/Beachlor/data/test/masks/image1765.png", source_mask);

    camera::PinholeCameraIntrinsic intrinsics;
    io::ReadIJsonConvertibleFromJSON("/home/hauke/code/Beachlor/data/intrinsics/intrinsics.json", intrinsics);
    auto intrinsics_matrix = core::eigen_converter::EigenMatrixToTensor(intrinsics.intrinsic_matrix_);

    //auto dxdy = source_color.RGBToGray().FilterSobelMaskout(source_mask.AsTensor().To(core::Bool));
    auto dxdy = source_color.RGBToGray().FilterSobel();
    auto f = dxdy.first.AsTensor();
    std::cout<< f.GetDtype().ToString()<<std::endl;
    f.Save("image.npy");

    auto mask = source_mask.To(core::Bool).PyrDownLogical();
    source_mask.AsTensor().Save("mask.npy");
    mask.AsTensor().Save("down_mask.npy");
    

    // std::vector<double> fitness;
    // std::vector<double> rsme;
    // t::pipelines::odometry::OdometryResult result;
    // core::Tensor init = core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0")));
    // for(int i = 0; i <20; i++ ){
    //     result = t::pipelines::odometry::RGBDOdometryMultiScale(
    //         t::geometry::RGBDImage(source_color, source_depth), 
    //         t::geometry::RGBDImage(target_color, target_depth),
    //         intrinsics_matrix, init, 
    //         1000.0f, 5.0f, {1}, t::pipelines::odometry::Method::PointToPlane, 
    //         t::pipelines::odometry::OdometryLossParams());
        
    //     init = result.transformation_;
    //     fitness.push_back(result.fitness_);
    //     rsme.push_back(result.inlier_rmse_);
    // }

    // core::Tensor fitness_t(std::move(fitness));
    // core::Tensor rsme_t(std::move(rsme));
    // fitness_t.Save("fitness_P2P.npy");
    // rsme_t.Save("rsme_P2P.npy");

    // auto t = t::pipelines::odometry::ComputeResidualMap(
    //     t::geometry::RGBDImage(source_color, source_depth),
    //     t::geometry::RGBDImage(target_color, target_depth),
    //     core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0")),
    //     intrinsics_matrix, t::pipelines::odometry::Method::Intensity, 
    //     1000.0f, 3.0f
    // );

    // t.AsTensor().Save("r_map.npy");

    // auto result = t::pipelines::odometry::RGBDMOdometryMultiScale(
    //     t::geometry::RGBDMImage(source_color, source_depth, source_mask), 
    //     t::geometry::RGBDMImage(target_color, target_depth, target_mask),
    //     intrinsics_matrix, core::Tensor::Eye(4, core::Float64, ((open3d::core::Device)("CPU:0"))), 
    //     1000.0f, 5.0f, {100, 100, 100}, t::pipelines::odometry::Method::PointToPlane, 
    //     t::pipelines::odometry::OdometryLossParams()

    // );

    // auto t_new = t::pipelines::odometry::ComputeResidualMap(
    //     t::geometry::RGBDImage(source_color, source_depth),
    //     t::geometry::RGBDImage(target_color, target_depth),
    //     result.transformation_,
    //     intrinsics_matrix, t::pipelines::odometry::Method::Intensity, 
    //     1000.0f, 3.0f
    // );
    // t_new.AsTensor().Save("r_mapt.npy");

    return 0;
};



