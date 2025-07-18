#include <open3d/core/Tensor.h>
#include <open3d/core/EigenConverter.h>
#include <open3d/camera/PinholeCameraIntrinsic.h>
#include <open3d/io/IJsonConvertibleIO.h>
#include <open3d/t/pipelines/odometry/RGBDMOdometry.h>
#include <open3d/t/pipelines/odometry/RGBDOdometry.h>


#include "test/test.h"
#include <filesystem>

using namespace open3d;
namespace fs =  std::filesystem;

int main(){
    
    //open3d::utility::Logger::GetInstance().SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    fs::path root_dir("/home/hauke/Downloads/rgbd_dataset_freiburg3_walking_rpy");
    core::Device device(core::Device::DeviceType::CPU, 0);

    test_masked_odometry(root_dir, 
        t::pipelines::odometry::Method::PointToPlane,
        t::pipelines::odometry::MaskMethod::SourceMask,
        device);

    return 0;
};

